"""
strategy_manager/signal.py — 因子信号 → 连续仓位转换

compute_target_positions_stateless: 无状态地将因子值转换为目标仓位。
使用 neutral band（中性区间）+ tanh 软压缩，输出 [-1, 1] 的连续仓位信号。

  position = sign(tanh(factor))           当 |factor| > band
  position = tanh(factor) / band * 0.5     当 |factor| <= band  (平滑过渡)

Neutral Band 逻辑：
  - |factor| < LOWER_BAND 时仓位为 0（信号太弱，不交易）
  - |factor| > UPPER_BAND 时仓位饱和（满仓多/空）
  - 中间区间线性过渡，避免频繁切换

2026-07-20 调整：
  LOWER_BAND 0.15→0.25：过滤更多噪声信号，减少高频无效交易
  UPPER_BAND 0.60→0.75：要求更强信号才满仓，降低换手率
"""
import torch

# 中性区间参数
# 2026-07-20: 提高阈值以减少高频交易中的手续费损耗
LOWER_BAND = 0.25    # 低于此值视为噪声，空仓（原 0.15）
UPPER_BAND = 0.75    # 高于此值满仓（原 0.60）


def compute_target_positions_stateless(factors: torch.Tensor) -> torch.Tensor:
    """无状态地将因子值转换为目标仓位序列。

    Args:
        factors: [N, T] 因子值张量

    Returns:
        [N, T] 目标仓位，值域 [-1, 1]
    """
    if not isinstance(factors, torch.Tensor):
        factors = torch.as_tensor(factors, dtype=torch.float32)

    # tanh 软压缩到 [-1, 1]
    raw = torch.tanh(factors)

    # Neutral band: 弱信号归零
    abs_raw = raw.abs()
    # 低于下限 → 0；高于上限 → 原值；中间 → 线性放大
    scale = torch.clamp((abs_raw - LOWER_BAND) / (UPPER_BAND - LOWER_BAND), 0.0, 1.0)
    position = raw * scale

    return position


def compute_target_positions_with_clamp(
    factors: torch.Tensor,
    max_position: float = 1.0,
    lower_band: float = LOWER_BAND,
    upper_band: float = UPPER_BAND,
) -> torch.Tensor:
    """带参数的目标仓位计算。"""
    raw = torch.tanh(factors)
    abs_raw = raw.abs()
    scale = torch.clamp((abs_raw - lower_band) / (upper_band - lower_band), 0.0, 1.0)
    position = raw * scale * max_position
    return position


def signal_to_action(signal: float) -> str:
    """将连续信号转为可读动作标签。"""
    if signal > 0.05:
        return f"做多 {signal:.1%}"
    elif signal < -0.05:
        return f"做空 {abs(signal):.1%}"
    return "空仓"
