"""
data_pipeline/timeframe_utils.py — OKX K 线周期工具

OKX bar 参数 → 人类可读周期 + 每年周期数（用于年化收益计算）。
"""
import math

# OKX bar 参数 → 标准周期映射
_OKX_BAR_MAP = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1H": 60, "2H": 120, "4H": 240, "6H": 360,
    "12H": 720, "1D": 1440, "1W": 10080, "1M": 43200,
}

# 每年分钟数（用于推算 periods_per_year）
_MINUTES_PER_YEAR = 365.25 * 24 * 60  # ≈ 525960


def parse_bar_to_minutes(bar: str) -> int:
    """OKX bar 字符串 → 分钟数。"""
    bar = str(bar).strip()
    if bar in _OKX_BAR_MAP:
        return _OKX_BAR_MAP[bar]
    # 尝试解析数字+单位
    for suffix, mult in [("m", 1), ("H", 60), ("h", 60), ("D", 1440), ("W", 10080)]:
        if bar.endswith(suffix):
            try:
                return int(bar[:-1]) * mult
            except ValueError:
                pass
    return 60  # 默认 1H


def infer_periods_per_year(time_array, default: int = 6240) -> int:
    """从时间序列推断每年的周期数。

    Args:
        time_array: 时间戳序列（秒级 epoch 或 datetime）
        default: 推断失败时的默认值

    Returns:
        每年周期数（用于 Sortino/Calmar 年化）
    """
    try:
        import numpy as np
        arr = np.asarray(time_array)
        if arr.ndim != 1 or len(arr) < 3:
            return default

        # 尝试解析为 epoch 秒
        if arr.dtype.kind in ('i', 'u', 'f'):
            diffs = np.diff(arr.astype(float))
            median_diff = float(np.median(diffs))
            if median_diff <= 0:
                return default
            # 如果差值是毫秒级，转秒
            if median_diff > 1e10:
                median_diff /= 1000.0
            periods_per_year = _MINUTES_PER_YEAR / (median_diff / 60.0)
            return int(round(periods_per_year))
        return default
    except Exception:
        return default


def bar_to_label(bar: str) -> str:
    """bar 参数 → 人类可读标签。"""
    return bar
