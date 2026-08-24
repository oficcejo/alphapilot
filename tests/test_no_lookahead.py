"""
tests/test_no_lookahead.py — 未来函数防泄漏回归测试

背景：旧版存在两处 look-ahead 泄漏（移植自上游 AlphaMaster 的修复）：
  1. target_ret 用 log(close[t]/close[t-1])（本 bar 收益），信号与收益
     同由 close[t] 决定 → 纯 RET 公式回测恒盈利、实盘失效。
     修复后 target_ret[t] = log(open[t+2]/open[t+1])（下一开盘成交）。
  2. vm._normalize_output 用全序列 mean/std 归一化（含未来）。
     修复后为滚动因果 z-score（窗口 500）。

这些测试锁死修复，防止回归。
"""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _synthetic_raw(T: int = 600, seed: int = 7) -> dict:
    """合成随机游走 OHLCV（open[t] = close[t-1]），[N=1, T]。"""
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(T, generator=g) * 0.002
    close = 100.0 * torch.exp(torch.cumsum(steps, dim=0))
    open_ = torch.cat([close[:1], close[:-1]])
    high = torch.maximum(open_, close) * 1.001
    low = torch.minimum(open_, close) * 0.999
    return {
        "close": close.unsqueeze(0).float(),
        "open": open_.unsqueeze(0).float(),
        "high": high.unsqueeze(0).float(),
        "low": low.unsqueeze(0).float(),
        "volume": torch.ones(1, T).float(),
    }


# ── 1. target_ret 对齐性 ────────────────────────────────────────────────

def test_target_ret_is_next_open():
    """target_ret[t] 必须等于 log(open[t+2]/open[t+1])，末两位为 0。"""
    from api.services.training_service import DataManager

    raw = _synthetic_raw(T=300)
    dm = DataManager(raw)
    o = raw["open"][0]
    tr = dm.target_ret[0]
    T = o.shape[0]

    for t in (0, 1, 100, T - 3):
        expected = math.log(o[t + 2].item() / o[t + 1].item())
        assert abs(tr[t].item() - expected) < 1e-6, f"t={t}: {tr[t]} != {expected}"

    assert tr[T - 2].item() == 0.0 and tr[T - 1].item() == 0.0


def test_target_ret_no_same_bar_correlation_for_ret_formula():
    """纯 RET 因子的仓位与 target_ret 同索引相关应≈0（随机游写下无泄漏）。"""
    from api.services.training_service import DataManager
    from model.vm import StackVM

    raw = _synthetic_raw(T=1500, seed=11)
    dm = DataManager(raw)

    vm = StackVM()
    factor = vm.execute([0], dm.feat_tensor)  # token 0 = RET
    assert factor is not None

    x = factor[0] - factor[0].mean()
    y = dm.target_ret[0]
    corr = ((x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False))).item()
    # 随机游走下理论值 0；泄漏口径下该值会显著为正（~0.5+）
    assert abs(corr) < 0.15, f"RET 因子与 target_ret 相关异常偏高: {corr}"


# ── 2. VM 归一化因果性 ──────────────────────────────────────────────────

def test_vm_normalize_output_causal():
    """截断序列的归一化输出必须与全序列输出在重叠段完全一致（无未来依赖）。"""
    from model.vm import StackVM

    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 1200, generator=g) * 5.0

    y_full = StackVM._normalize_output(x)
    cut = 900
    y_cut = StackVM._normalize_output(x[:, :cut])

    assert torch.allclose(y_full[:, :cut], y_cut, atol=1e-5), (
        "归一化输出依赖了未来数据（截断后重叠段不一致）"
    )


def test_vm_normalize_output_warmup_zero():
    """滚动窗口 warm-up 期（前 W-1 根）输出必须为 0。"""
    from model.vm import StackVM

    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 800, generator=g) * 3.0
    y = StackVM._normalize_output(x)
    assert (y[:, :499] == 0).all()


# ── 3. 端到端：纯 RET 公式在新口径下不应盈利 ────────────────────────────

def test_pure_ret_formula_cannot_profit():
    """同一纯 RET 公式：泄漏口径显著盈利（复现旧 bug），修复口径不盈利。"""
    from model.features import MT5FeatureEngineer
    from model.vm import StackVM
    from strategy_manager.signal import compute_target_positions_stateless

    raw = _synthetic_raw(T=1500, seed=11)
    feat = MT5FeatureEngineer.compute_features(raw)
    vm = StackVM()
    factor = vm.execute([0], feat)
    pos = compute_target_positions_stateless(factor)

    o = raw["open"][0]
    c = raw["close"][0]
    T = o.shape[0]

    turnover = torch.abs(pos - torch.roll(pos, 1, dims=1))
    turnover[:, 0] = 0.0

    # 修复口径（下一开盘成交）
    fwd = torch.zeros(1, T)
    denom = o[1:-1].clone()
    denom[denom == 0] = 1.0
    fwd[:, : T - 2] = torch.log(o[2:] / denom)
    pnl_fixed = (pos * fwd - turnover * 0.0008).mean().item()

    # 旧泄漏口径（本 bar 收盘收益）：pos[t]*ret[t] >= 0 恒成立
    same_bar = torch.zeros(1, T)
    same_bar[:, 1:] = torch.log(c[1:] / c[:-1])
    pnl_leak = (pos * same_bar).mean().item()

    assert pnl_leak > 1e-6, "泄漏口径应为正（否则测试本身失效）"
    assert pnl_fixed < pnl_leak * 0.1, (
        f"修复口径均值 {pnl_fixed:.6f} 应远小于泄漏口径 {pnl_leak:.6f}"
    )
