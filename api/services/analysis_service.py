"""
api/services/analysis_service.py — 实时分析服务

从 OKX / MT5 / TradingView 获取 K 线，计算最新因子信号和仓位建议。
"""
import time
import numpy as np
import torch
from typing import Optional

from config import Config
from model.vocab import FORMULA_VOCAB
from model.vm import StackVM
from model.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless, signal_to_action
from data_pipeline.okx_client import OKXClient, get_public_client
from data_pipeline.parquet_manager import load_parquet_to_raw_dict
from api.services.strategy_service import load_strategy, decode_formula, eval_strategy_factor


class AnalysisService:
    """实时分析服务。"""

    def __init__(self):
        self.vm = StackVM()

    @staticmethod
    def _sanitize_candles(candles: list[list]) -> list[list]:
        """清洗 OKX K 线列表：过滤未收盘 bar + 按时间升序排序。

        OKX 返回按时间倒序且首根为进行中 bar；特征计算必须使用
        升序、已收盘的 K 线（与训练/回测口径一致）。
        """
        cleaned = {}
        for c in candles:
            if not c or len(c) < 6:
                continue
            ts = int(c[0])
            if ts in cleaned:
                continue
            # confirm: '0'=未收盘（进行中），'1'=已收盘
            if len(c) >= 9 and str(c[8]) == "0":
                continue
            cleaned[ts] = c
        return [cleaned[ts] for ts in sorted(cleaned)]

    def _candles_to_raw_dict(self, candles: list[list], symbol: str, bar: str) -> dict:
        """OKX K 线列表 → raw_dict（输入须为升序已收盘 K 线）。"""
        if not candles:
            raise ValueError("K 线数据为空")
        close = np.array([float(c[4]) for c in candles], dtype=np.float64)
        open_ = np.array([float(c[1]) for c in candles], dtype=np.float64)
        high = np.array([float(c[2]) for c in candles], dtype=np.float64)
        low = np.array([float(c[3]) for c in candles], dtype=np.float64)
        volume = np.array([float(c[5]) for c in candles], dtype=np.float64)
        time_arr = np.array([int(c[0]) // 1000 if int(c[0]) > 1e12 else int(c[0]) for c in candles], dtype=np.float64)

        return {
            "close": torch.from_numpy(close).unsqueeze(0).float(),
            "open": torch.from_numpy(open_).unsqueeze(0).float(),
            "high": torch.from_numpy(high).unsqueeze(0).float(),
            "low": torch.from_numpy(low).unsqueeze(0).float(),
            "volume": torch.from_numpy(volume).unsqueeze(0).float(),
            "time": time_arr,
        }

    @staticmethod
    def _forward_open_ret(raw_dict: dict) -> "torch.Tensor":
        """下一开盘成交的对数收益 target_ret[t] = log(open[t+2]/open[t+1])。

        与训练/回测口径一致，最后两个时间步置 0。
        """
        open_ = raw_dict["open"]
        t = open_.shape[1]
        fwd = torch.zeros_like(open_)
        if t >= 3:
            denominator = open_[:, 1:-1].clone()
            denominator[denominator == 0] = 1.0
            fwd[:, : t - 2] = torch.log(open_[:, 2:] / denominator)
        return fwd

    def analyze_okx(
        self,
        strategy_path: str,
        inst_id: str,
        bar: str = "1H",
        limit: int = None,
    ) -> dict:
        """从 OKX 获取最新 K 线并计算信号。

        Args:
            strategy_path: 策略 JSON 路径
            inst_id: OKX 合约 ID，如 BTC-USDT-SWAP
            bar: K 线周期
            limit: K 线数量（默认取 Config.REALTIME_MIN_BARS，保证归一化收敛）

        Returns:
            分析结果（含最新信号、仓位建议、因子值序列）
        """
        # 1. 加载策略（支持单因子策略与组合策略）
        strategy = load_strategy(strategy_path)
        formula = strategy.get("formula")
        formula_decoded = strategy.get("formula_decoded", "")

        # 2. 获取 OKX 已收盘 K 线（自动分页、升序、过滤未收盘 bar）
        min_bars = getattr(Config, "REALTIME_MIN_BARS", 800)
        need = max(int(limit or 0), min_bars)
        client = get_public_client()
        candles = client.get_recent_candles(inst_id, bar, total=need, only_confirmed=True)
        if not candles:
            raise RuntimeError(f"未获取到 {inst_id} K线数据")
        if len(candles) < min_bars:
            return {
                "source": "okx",
                "inst_id": inst_id,
                "bar": bar,
                "state": "insufficient",
                "n_candles": len(candles),
                "message": (
                    f"历史 bar 不足（{len(candles)}/{min_bars}），"
                    f"无法稳定计算特征与滚动归一化"
                ),
            }

        raw_dict = self._candles_to_raw_dict(candles, inst_id, bar)

        # 3. 计算特征
        feat = MT5FeatureEngineer.compute_features(raw_dict)

        # 4. 执行因子求值（兼容单策略与多因子组合）
        with torch.no_grad():
            factor = eval_strategy_factor(strategy, self.vm, feat)
        if factor is None:
            raise ValueError("因子计算失败（无效策略或求值异常）")

        # 5. 计算仓位
        position = compute_target_positions_stateless(factor)

        # 6. 最新信号（最后一根已收盘 bar）
        last_factor = float(factor[0, -1].item())
        last_position = float(position[0, -1].item())
        last_price = float(raw_dict["close"][0, -1].item())
        last_time = int(raw_dict["time"][-1])

        # 7. 最近 100 根的信号序列
        n_show = min(100, factor.shape[1])
        factor_recent = factor[0, -n_show:].numpy().tolist()
        position_recent = position[0, -n_show:].numpy().tolist()
        close_recent = raw_dict["close"][0, -n_show:].numpy().tolist()
        time_recent = raw_dict["time"][-n_show:].tolist()

        # 8. 信号统计
        long_bars = int((position[0] > 0.05).sum().item())
        short_bars = int((position[0] < -0.05).sum().item())
        flat_bars = int(factor.shape[1] - long_bars - short_bars)

        # 9. PnL 估算（无成本，下一开盘成交口径，与回测一致）
        fwd_ret = self._forward_open_ret(raw_dict)
        pnl = (position * fwd_ret)
        cum_pnl = pnl.reshape(-1).cumsum(0).numpy()
        total_ret = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0

        return {
            "source": "okx",
            "inst_id": inst_id,
            "bar": bar,
            "n_candles": len(candles),
            "strategy": {
                "formula": formula,
                "formula_decoded": formula_decoded,
                "symbol": strategy.get("symbol"),
            },
            "latest": {
                "time": last_time,
                "price": round(last_price, 6),
                "factor": round(last_factor, 4),
                "position": round(last_position, 4),
                "action": signal_to_action(last_position),
            },
            "signal_stats": {
                "long_bars": long_bars,
                "short_bars": short_bars,
                "flat_bars": flat_bars,
                "long_pct": round(long_bars / max(factor.shape[1], 1) * 100, 1),
                "short_pct": round(short_bars / max(factor.shape[1], 1) * 100, 1),
            },
            "series": {
                "time": time_recent,
                "close": [round(c, 6) for c in close_recent],
                "factor": [round(f, 4) for f in factor_recent],
                "position": [round(p, 4) for p in position_recent],
            },
            "estimated_return": round(total_ret * 100, 2),
        }

    def analyze_parquet(
        self,
        strategy_path: str,
        data_file: str,
    ) -> dict:
        """从本地 Parquet 分析信号（MT5 / 本地数据模式）。"""
        strategy = load_strategy(strategy_path)
        formula = strategy.get("formula")
        formula_decoded = strategy.get("formula_decoded", "")

        raw_dict = load_parquet_to_raw_dict(data_file)
        feat = MT5FeatureEngineer.compute_features(raw_dict)

        with torch.no_grad():
            factor = eval_strategy_factor(strategy, self.vm, feat)
        if factor is None:
            raise ValueError("因子计算失败（无效策略或求值异常）")

        position = compute_target_positions_stateless(factor)

        last_factor = float(factor[0, -1].item())
        last_position = float(position[0, -1].item())
        last_price = float(raw_dict["close"][0, -1].item())

        n_show = min(100, factor.shape[1])
        factor_recent = factor[0, -n_show:].numpy().tolist()
        position_recent = position[0, -n_show:].numpy().tolist()
        close_recent = raw_dict["close"][0, -n_show:].numpy().tolist()
        time_arr = raw_dict.get("time", np.arange(factor.shape[1]))
        time_recent = time_arr[-n_show:].tolist() if hasattr(time_arr, 'tolist') else list(time_arr)[-n_show:]

        long_bars = int((position[0] > 0.05).sum().item())
        short_bars = int((position[0] < -0.05).sum().item())

        fwd_ret = self._forward_open_ret(raw_dict)
        pnl = (position * fwd_ret)
        cum_pnl = pnl.reshape(-1).cumsum(0).numpy()
        total_ret = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0

        return {
            "source": "parquet",
            "data_file": data_file,
            "n_candles": factor.shape[1],
            "strategy": {
                "formula": formula,
                "formula_decoded": formula_decoded,
                "symbol": strategy.get("symbol"),
            },
            "latest": {
                "price": round(last_price, 6),
                "factor": round(last_factor, 4),
                "position": round(last_position, 4),
                "action": signal_to_action(last_position),
            },
            "signal_stats": {
                "long_bars": long_bars,
                "short_bars": short_bars,
                "flat_bars": int(factor.shape[1] - long_bars - short_bars),
                "long_pct": round(long_bars / max(factor.shape[1], 1) * 100, 1),
                "short_pct": round(short_bars / max(factor.shape[1], 1) * 100, 1),
            },
            "series": {
                "time": time_recent,
                "close": [round(c, 6) for c in close_recent],
                "factor": [round(f, 4) for f in factor_recent],
                "position": [round(p, 4) for p in position_recent],
            },
            "estimated_return": round(total_ret * 100, 2),
        }


# 全局单例
analysis_service = AnalysisService()
