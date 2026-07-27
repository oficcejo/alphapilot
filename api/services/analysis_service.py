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
from api.services.strategy_service import load_strategy, decode_formula


class AnalysisService:
    """实时分析服务。"""

    def __init__(self):
        self.vm = StackVM()

    def _candles_to_raw_dict(self, candles: list[list], symbol: str, bar: str) -> dict:
        """OKX K 线列表 → raw_dict。"""
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

    def analyze_okx(
        self,
        strategy_path: str,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
    ) -> dict:
        """从 OKX 获取最新 K 线并计算信号。

        Args:
            strategy_path: 策略 JSON 路径
            inst_id: OKX 合约 ID，如 BTC-USDT-SWAP
            bar: K 线周期
            limit: K 线数量

        Returns:
            分析结果（含最新信号、仓位建议、因子值序列）
        """
        # 1. 加载策略
        strategy = load_strategy(strategy_path)
        formula = strategy.get("formula")
        if not formula:
            raise ValueError("策略文件中无公式")
        formula_decoded = strategy.get("formula_decoded", decode_formula(formula))

        # 2. 获取 OKX K 线
        client = get_public_client()
        candles = client.get_candles(inst_id, bar, limit)
        if not candles:
            raise RuntimeError(f"未获取到 {inst_id} K线数据")

        raw_dict = self._candles_to_raw_dict(candles, inst_id, bar)

        # 3. 计算特征
        feat = MT5FeatureEngineer.compute_features(raw_dict)

        # 4. 执行公式
        with torch.no_grad():
            factor = self.vm.execute(formula, feat)
        if factor is None:
            raise ValueError("公式执行失败")

        # 5. 计算仓位
        position = compute_target_positions_stateless(factor)

        # 6. 最新信号
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

        # 9. 简单 PnL 估算（无成本）
        close = raw_dict["close"]
        eps = 1e-9
        log_ret = torch.zeros_like(close)
        log_ret[:, 1:] = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        pnl = (position * log_ret)
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
        if not formula:
            raise ValueError("策略文件中无公式")
        formula_decoded = strategy.get("formula_decoded", decode_formula(formula))

        raw_dict = load_parquet_to_raw_dict(data_file)
        feat = MT5FeatureEngineer.compute_features(raw_dict)

        with torch.no_grad():
            factor = self.vm.execute(formula, feat)
        if factor is None:
            raise ValueError("公式执行失败")

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

        close = raw_dict["close"]
        eps = 1e-9
        log_ret = torch.zeros_like(close)
        log_ret[:, 1:] = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        pnl = (position * log_ret)
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
