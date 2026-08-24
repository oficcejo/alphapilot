"""
rigorous_backtest_audit.py — 独立严谨回测审计（移植自上游 AlphaMaster）

不复用训练 reward，从第一性原理计算 PnL（下一开盘成交口径）：
  target_ret[t] = log(open[t+2] / open[t+1])
  pnl[t] = position[t] * target_ret[t] - |Δposition[t]| * cost

审计维度：
  - 全样本 / 前半段 / 后半段 / 尾段 20%（近似 OOS）
  - 成本压力测试（1x / 2x / 3x）
  - 暴露度、换手率、多空比例（识别 Beta 因子与稀疏交易）

用法：
  python rigorous_backtest_audit.py                       # 审计 strategies/ 全部策略
  python rigorous_backtest_audit.py --strategy xxx.json   # 指定单个策略
  python rigorous_backtest_audit.py --strategy a.json --data b.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from config import Config
from model.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless
from api.services.strategy_service import load_strategy, eval_strategy_factor
from data_pipeline.parquet_manager import load_parquet_to_raw_dict
from data_pipeline.timeframe_utils import infer_periods_per_year
from model.features import MT5FeatureEngineer

COST_RATE = 0.0005   # 手续费
SLIPPAGE = 0.0003    # 滑点


def find_data_file(strategy: dict) -> str | None:
    """按策略元数据定位数据文件：data_file 字段 → data/{symbol}_{tf}.parquet。"""
    df = strategy.get("data_file")
    if df and pathlib.Path(df).exists():
        return df
    symbol = strategy.get("symbol")
    tf = strategy.get("timeframe") or "1H"
    if symbol:
        cand = pathlib.Path(Config.DATA_DIR) / f"{symbol}_{tf}.parquet"
        if cand.exists():
            return str(cand)
        cand = pathlib.Path(Config.DATA_DIR) / f"{symbol}.parquet"
        if cand.exists():
            return str(cand)
    return None


def forward_open_ret(raw_dict: dict) -> torch.Tensor:
    """target_ret[t] = log(open[t+2]/open[t+1])，末两位置 0。"""
    open_ = raw_dict["open"]
    t = open_.shape[1]
    fwd = torch.zeros_like(open_)
    if t >= 3:
        denom = open_[:, 1:-1].clone()
        denom[denom == 0] = 1.0
        fwd[:, : t - 2] = torch.log(open_[:, 2:] / denom)
    return fwd


def max_drawdown(pnl: np.ndarray) -> float:
    equity = np.exp(np.cumsum(np.clip(pnl, -0.5, 0.5)))
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max())


def sortino(pnl: np.ndarray, ppy: int) -> float:
    down = pnl[pnl < 0]
    ds = down.std() if len(down) > 0 else pnl.std()
    ds = max(ds, pnl.std() * 0.2, 1e-9)
    return float(pnl.mean() / ds * math.sqrt(ppy))


def sharpe(pnl: np.ndarray, ppy: int) -> float:
    return float(pnl.mean() / (pnl.std() + 1e-12) * math.sqrt(ppy))


def segment_metrics(pnl: np.ndarray, pos_flat: np.ndarray,
                    turnover_mean: float, ppy: int) -> dict:
    total = float(np.sum(pnl))
    years = max(len(pnl) / ppy, 1e-9)
    cagr = (math.exp(total) ** (1.0 / years) - 1.0) if total > -1 else -1.0
    wins = (pnl > 0).sum()
    active = (pnl != 0).sum()
    return {
        "bars": int(len(pnl)),
        "total_return_pct": round((math.exp(total) - 1.0) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sortino": round(sortino(pnl, ppy), 3),
        "sharpe": round(sharpe(pnl, ppy), 3),
        "max_drawdown_pct": round(max_drawdown(pnl) * 100, 2),
        "win_rate_pct": round(wins / max(active, 1) * 100, 1),
        "exposure_pct": round(float(np.abs(pos_flat).mean()) * 100, 1),
        "avg_turnover": round(float(turnover_mean), 4),
    }


def audit_strategy(path: str, data_file: str | None,
                   leverage: int = 1) -> dict | None:
    strategy = load_strategy(path)
    if not data_file:
        data_file = find_data_file(strategy)
    if not data_file:
        return {"strategy": path, "error": "未找到匹配的数据文件"}

    raw_dict = load_parquet_to_raw_dict(data_file)
    feat = MT5FeatureEngineer.compute_features(raw_dict)
    vm = StackVM()
    with torch.no_grad():
        factor = eval_strategy_factor(strategy, vm, feat)
    if factor is None:
        return {"strategy": path, "error": "因子求值失败"}

    close_t = raw_dict["time"]
    ppy = infer_periods_per_year(close_t, default=6240)

    pos = compute_target_positions_stateless(factor)
    prev_pos = torch.roll(pos, 1, dims=1)
    prev_pos[:, 0] = 0.0
    turnover = torch.abs(pos - prev_pos)

    fwd = forward_open_ret(raw_dict)
    T = fwd.shape[1]

    long_ratio = float((pos > 0.05).float().mean())
    short_ratio = float((pos < -0.05).float().mean())

    result: dict = {
        "strategy": pathlib.Path(path).name,
        "data": pathlib.Path(data_file).name,
        "bars": T,
        "periods_per_year": ppy,
        "long_pct": round(long_ratio * 100, 1),
        "short_pct": round(short_ratio * 100, 1),
        "segments": {},
        "cost_stress_last20": {},
    }

    segments = {
        "full": (0, T),
        "first_half": (0, T // 2),
        "second_half": (T // 2, T),
        "last_20pct": (int(T * 0.8), T),
    }
    for name, (a, b) in segments.items():
        pnl = ((pos * fwd - turnover * (COST_RATE + SLIPPAGE)) * leverage)[:, a:b]
        result["segments"][name] = segment_metrics(
            pnl.reshape(-1).numpy(), pos[:, a:b].reshape(-1).numpy(),
            float(turnover[:, a:b].mean()), ppy,
        )

    # 成本压力测试（尾段 20%）
    a, b = segments["last_20pct"]
    for mult in (1, 2, 3):
        pnl = ((pos * fwd - turnover * (COST_RATE + SLIPPAGE) * mult) * leverage)[:, a:b]
        flat = pnl.reshape(-1).numpy()
        result["cost_stress_last20"][f"{mult}x"] = {
            "total_return_pct": round((math.exp(flat.sum()) - 1.0) * 100, 2),
            "sortino": round(sortino(flat, ppy), 3),
        }

    # 风险信号
    flags = []
    seg = result["segments"]
    if seg["full"]["exposure_pct"] < 10:
        flags.append("暴露度过低（<10%），可能为稀疏偶发交易刷分")
    if seg["first_half"]["total_return_pct"] <= 0 or seg["second_half"]["total_return_pct"] <= 0:
        flags.append("前后半段收益不同号为正：策略只在特定市场环境有效")
    if max(long_ratio, short_ratio) > 0.85:
        flags.append("单边占比 >85%：疑似 Beta 因子而非 Alpha 因子")
    if seg["last_20pct"]["sortino"] <= 0:
        flags.append("尾段 20% Sortino ≤ 0：近期已失效")
    result["flags"] = flags
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="独立严谨回测审计（下一开盘成交口径）")
    parser.add_argument("--strategy", default=None, help="指定策略 JSON 路径")
    parser.add_argument("--data", default=None, help="指定 Parquet 数据路径")
    parser.add_argument("--leverage", type=int, default=1, help="杠杆（默认 1）")
    args = parser.parse_args()

    if args.strategy:
        paths = [args.strategy]
    else:
        sdir = pathlib.Path(Config.STRATEGIES_DIR)
        paths = sorted(str(p) for p in sdir.glob("*.json"))
    if not paths:
        print("未找到任何策略文件")
        return

    reports = []
    for p in paths:
        try:
            r = audit_strategy(p, args.data, args.leverage)
        except Exception as e:
            r = {"strategy": pathlib.Path(p).name, "error": f"{type(e).__name__}: {e}"}
        reports.append(r)

    for r in reports:
        print("=" * 78)
        if "error" in r:
            print(f"  {r['strategy']}: {r['error']}")
            continue
        print(f"  {r['strategy']}  @  {r['data']}  ({r['bars']} bars, {r['periods_per_year']}/yr)")
        print(f"  多空比例: {r['long_pct']}% / {r['short_pct']}%")
        hdr = f"  {'段':<12}{'总收益%':>10}{'CAGR%':>10}{'Sortino':>9}{'Sharpe':>8}{'MaxDD%':>8}{'胜率%':>7}{'暴露%':>7}"
        print(hdr)
        for name, m in r["segments"].items():
            print(f"  {name:<12}{m['total_return_pct']:>10}{m['cagr_pct']:>10}"
                  f"{m['sortino']:>9}{m['sharpe']:>8}{m['max_drawdown_pct']:>8}"
                  f"{m['win_rate_pct']:>7}{m['exposure_pct']:>7}")
        print("  成本压力(尾段20%): " + "  ".join(
            f"{k}: {v['total_return_pct']}%" for k, v in r["cost_stress_last20"].items()))
        for f in r.get("flags", []):
            print(f"  [!] {f}")

    out = pathlib.Path("rigorous_audit_report.json")
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 78)
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
