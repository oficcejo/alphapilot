import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .vocab import FORMULA_VOCAB


class NewtonSchulzLowRankDecay:
    """
    Low-Rank Decay (LoRD) using Newton-Schulz iteration.
    
    A more efficient regularization method that targets low-rank structure
    in attention and key parameters. Uses Newton-Schulz iteration to compute
    the minimum singular vectors without explicit SVD.
    
    Args:
        named_parameters: Model's named parameters
        decay_rate: Strength of low-rank decay
        num_iterations: Number of Newton-Schulz iterations (default: 5)
        target_keywords: If specified, only decay parameters matching these keywords
    """
    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self):
        """Apply Newton-Schulz low-rank decay to attention parameters."""
        for name, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            # Transpose if needed for efficiency
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            # Normalize by spectral norm
            norm = X.norm() + 1e-8
            X = X / norm
            
            # Initialize Y for Newton-Schulz iteration
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
            # This converges to the orthogonal matrix with same singular vectors
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            # Apply low-rank decay
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    """Monitor the effective rank (stable rank) of model parameters."""
    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []
    
    @torch.no_grad()
    def compute(self):
        """Compute average stable rank of target parameters."""
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key Normalization for Attention"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))
    
    def forward(self, q, k):
        # Normalize Q and K independently
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    """Swish GLU activation function"""
    def __init__(self, d_in, d_ff):
        super().__init__()
        # P2-2: Separate gate/up projections — avoids allocating a 2×d_ff
        # intermediate tensor and subsequent chunk. For d_in=96, d_ff=192
        # the saving is small but the code is cleaner.
        self.w_gate = nn.Linear(d_in, d_ff)
        self.w_up = nn.Linear(d_in, d_ff)
        self.fc = nn.Linear(d_ff, d_in)

    def forward(self, x):
        return self.fc(self.w_up(x) * F.silu(self.w_gate(x)))


class MTPHead(nn.Module):
    """Multi-Task Pooling Head for multi-objective learning"""
    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
        ])
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks)
        )
    
    def forward(self, x):
        # Route to appropriate task heads
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        
        # Compute all task outputs
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)  # [B, num_tasks, vocab_size]
        
        # Weighted combination
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
        return weighted, task_probs


class QKNormAttention(nn.Module):
    """Multi-Head Attention with integrated QK-Normalization.

    Replaces nn.MultiheadAttention + QKNorm (which was defined but never
    called in forward). This class properly normalizes Q and K to unit L2
    norm before scaling, stabilizing attention scores for small d_model.

    Parameter naming (in_proj/out_proj/qk_norm) is compatible with
    NewtonSchulzLowRankDecay and StableRankMonitor target_keywords.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} not divisible by nhead={nhead}"
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead

        # Fused QKV projection (named 'in_proj' for LoRD/StableRank compatibility)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        # Output projection (named 'out_proj' for compatibility)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # QK-Normalization: learnable per-dimension scale.
        # L2-normalized Q·K^T gives cosine similarity ∈ [-1, 1];
        # qk_norm^2 controls effective temperature. init=1.0 → reasonable softmax.
        self.qk_norm = nn.Parameter(torch.ones(1, 1, 1, self.head_dim))

        self.dropout = nn.Dropout(dropout)

        # P2-1: Cache causal masks — during autoregressive sampling the model is
        # called with T=1..MAX_FORMULA_LEN, so ~8 unique mask sizes × 3 layers ×
        # 3 loops × 2 (new+elite) = ~144 mask allocations per step. Caching avoids
        # all of them.
        self._causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Return cached upper-triangular boolean causal mask for length T."""
        key = (T, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
        return self._causal_mask_cache[key]

    def forward(self, x: torch.Tensor, is_causal: bool = True,
                **kwargs) -> tuple[torch.Tensor, None]:
        """Self-attention: query=key=value=x.

        Args:
            x: [B, T, d_model]
            is_causal: if True, apply causal mask (position i attends to j<=i)

        Returns:
            (output, None) — output is [B, T, d_model]; None for attn_weights
        """
        B, T, C = x.shape

        # Fused QKV projection
        qkv = self.in_proj(x)                       # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)              # each [B, T, C]

        # Reshape to multi-head: [B, nhead, T, head_dim]
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # QK-Normalization: L2 normalize, then apply learnable scale
        q = F.normalize(q, p=2, dim=-1) * self.qk_norm
        k = F.normalize(k, p=2, dim=-1) * self.qk_norm

        # Scaled dot-product attention.
        # After QK-norm: q@k^T = cosine_sim * qk_norm^2 ∈ [-scale^2, scale^2].
        # Additional 1/sqrt(head_dim) is the standard attention scaling.
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if is_causal:
            mask = self._get_causal_mask(T, x.device)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)                 # [B, nhead, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out, None


class LoopedTransformerLayer(nn.Module):
    """Looped Transformer Layer - recurrent processing within a layer"""
    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead

        # QK-Norm Attention (replaces nn.MultiheadAttention + dead QKNorm)
        self.attention = QKNormAttention(d_model, nhead, dropout=dropout)

        # RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # SwiGLU FFN instead of standard FFN
        self.ffn = SwiGLU(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_causal=True):
        # Looped processing - recurrent refinement
        for _ in range(self.num_loops):
            # Self-attention with residual
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, is_causal=is_causal)
            x = x + self.dropout(attn_out)

            # FFN with residual
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)

        return x


class LoopedTransformer(nn.Module):
    """Looped Transformer Encoder with multiple loop iterations"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, is_causal=True):
        for layer in self.layers:
            x = layer(x, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # d_model 64→96：vocab 剪枝后为 ~94，96 维 embedding 足以支撑，
        # 兼顾容量与 CPU 自回归采样的速度（128 维每步耗时过高）。
        self.d_model = 96
        self.features_list = list(FORMULA_VOCAB.feature_names)
        self.ops_list = list(FORMULA_VOCAB.operator_names)
        
        self.vocab = list(FORMULA_VOCAB.token_names)
        self.vocab_size = FORMULA_VOCAB.size
        
        # Embedding
        # pos_emb 用固定上限 20，与 MAX_FORMULA_LEN 解耦：
        # - 阶段A (len=8) 和阶段B (len=14) 都能用同一模型权重
        # - 测试无需随配置变更而调整
        # - 手工评估或 14-token 公式都在范围内
        _POS_EMB_MAX = 20
        self._max_seq = _POS_EMB_MAX
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        # P1-2: 小随机初始化（原零初始化导致训练初期无位置信息）
        self.pos_emb = nn.Parameter(torch.randn(1, _POS_EMB_MAX, self.d_model) * 0.02)
        
        # Enhanced Transformer with Looped Transformer
        # num_layers 2→3、dim_feedforward 128→192：配合 d_model=96 适度扩容。
        # 4 层 looped(×3 loops) 在 CPU 自回归采样下每步耗时过高，取 3 层平衡。
        # nhead=4 → head_dim=24。
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=4,
            num_layers=3,
            dim_feedforward=192,
            num_loops=3,
            dropout=0.1
        )
        
        # RMSNorm instead of LayerNorm
        self.ln_f = RMSNorm(self.d_model)
        
        # P1-1: Weight-tied output head (replaces MTPHead — 3 heads had no
        # differentiated supervision and task_probs was discarded by engine).
        # Weight tying reuses token_emb for output projection, saving ~38K params.
        self.head_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        if T > self._max_seq:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq {self._max_seq}. "
                f"Increase ModelConfig.MAX_FORMULA_LEN."
            )
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]

        # Custom QKNormAttention handles causal masking internally
        x = self.blocks(x, is_causal=True)
        x = self.ln_f(x)
        
        last_emb = x[:, -1, :]

        # P1-1: Weight-tied logits (reuses token embedding as output projection)
        logits = last_emb @ self.token_emb.weight.T + self.head_bias
        value = self.head_critic(last_emb)

        return logits, value, None
