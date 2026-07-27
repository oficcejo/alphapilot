"""
api/services/backtest_service.py — 策略回测服务

加载策略 + 数据，执行回测，生成资金曲线和绩效报告。
"""
import json
import math
import pathlib
import numpy as np
import torch
from typing import Optional

from config import Config
from model.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model.vm import StackVM
from model.features import MT5FeatureEngineer
from model.backtest import MT5Backtest
from strategy_manager.signal import compute_target_positions_stateless
from data_pipeline.parquet_manager import load_parquet_to_raw_dict
from data_pipeline.timeframe_utils import infer_periods_per_year
from api.services.strategy_service import load_strategy, decode_formula


class BacktestService:
    """策略回测服务。"""

    def __init__(self):
        self.vm = StackVM()

    def run_backtest(
        self,
        strategy_path: str,
        data_file: str,
        cost_rate: float = 0.0005,
        slippage: float = 0.0003,
        initial_capital: float = 10000.0,
        leverage: int = 5,
    ) -> dict:
        """执行完整回测。

        Args:
            strategy_path: 策略 JSON 路径
            data_file: Parquet 数据文件路径
            cost_rate: 手续费率
            slippage: 滑点
            initial_capital: 初始资金
            leverage: 杠杆

        Returns:
            回测结果（含资金曲线、绩效指标、交易明细）
        """
        # 1. 加载策略
        strategy = load_strategy(strategy_path)
        formula = strategy.get("formula")
        if not formula:
            raise ValueError("策略文件中无公式")
        formula_decoded = strategy.get("formula_decoded", decode_formula(formula))

        # 2. 加载数据
        raw_dict = load_parquet_to_raw_dict(data_file)
        feat = MT5FeatureEngineer.compute_features(raw_dict)
        close = raw_dict["close"]  # [N, T]
        time_arr = raw_dict.get("time", np.arange(close.shape[1]))

        N, T = close.shape
        periods_per_year = infer_periods_per_year(time_arr, default=6240)

        # 3. 执行公式得到因子
        with torch.no_grad():
            factor = self.vm.execute(formula, feat)
        if factor is None:
            raise ValueError("公式执行失败（无效公式）")

        # 4. 计算仓位信号
        position = compute_target_positions_stateless(factor)  # [N, T]

        # 5. 计算 PnL（对数收益率空间）
        eps = 1e-9
        log_ret = torch.zeros_like(close)
        log_ret[:, 1:] = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        target_ret = log_ret

        # 换手率
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        # PnL = position * log_ret - turnover * cost
        # position * log_ret 是单 bar 对数收益率（百分比空间，非金额）
        total_cost = cost_rate + slippage
        # 杠杆放大仓位敞口，成本也相应放大
        bar_logret = (position * target_ret - turnover * total_cost) * leverage

        # 6. 资金曲线（对数收益率复合：equity = capital * exp(cumsum(logret))）
        bar_ret_flat = bar_logret.reshape(-1).numpy()
        # 限制单 bar 极端值，防数值溢出
        bar_ret_flat = np.clip(bar_ret_flat, -0.5, 0.5)
        cum_logret = np.cumsum(bar_ret_flat)
        equity = initial_capital * np.exp(cum_logret)
        equity = np.maximum(equity, 0)  # 不允许负数

        # 时间轴
        if hasattr(time_arr, '__len__'):
            time_list = time_arr.tolist() if hasattr(time_arr, 'tolist') else list(time_arr)
        else:
            time_list = list(range(T))

        # 采样：如果数据量太大，降采样到 500 点
        max_points = 500
        if T > max_points:
            step = T // max_points
            equity_sampled = equity[::step].tolist()
            time_sampled = [time_list[i] for i in range(0, T, step)]
        else:
            equity_sampled = equity.tolist()
            time_sampled = list(time_list)

        # ── 数据时长（用于判断年化指标是否可信）──
        data_duration_years = T / periods_per_year if periods_per_year > 0 else 0
        data_duration_days = data_duration_years * 365
        # 样本过短阈值：不足 90 天的年化指标不可信
        is_short_sample = data_duration_days < 90

        # ── 检测样本内偏差（策略训练品种 == 回测品种）──
        strat_symbol = strategy.get("symbol")
        # 从文件名推断数据品种
        data_filename = pathlib.Path(data_file).stem
        is_in_sample = False
        if strat_symbol and strat_symbol in data_filename:
            is_in_sample = True

        # 7. 绩效指标
        # bar_logret 是对数收益率，用于 Sortino/Sharpe 计算
        bt = MT5Backtest(cost_rate=total_cost, periods_per_year=periods_per_year)

        # Sortino/Calmar：获取截断前原始值，标记是否被 clip
        raw_sortino = bt._sortino(bar_logret).item()
        raw_calmar = bt._calmar(bar_logret).item()
        # _sortino clip 到 ±20，_calmar clip 到 ±10
        sortino_clipped = abs(raw_sortino) >= 19.99
        calmar_clipped = abs(raw_calmar) >= 9.99
        sortino = raw_sortino
        calmar = raw_calmar

        # 年化收益：使用 CAGR（几何复合年化）
        # CAGR = (1 + total_return)^(1/years) - 1
        total_return = float(equity[-1] / initial_capital - 1)
        if data_duration_years > 0 and total_return > -1:
            cagr = float(math.pow(1 + total_return, 1.0 / data_duration_years) - 1)
        else:
            cagr = 0.0
        # 线性年化（仅用于对比参考）
        ann_ret_linear = float(bar_logret.mean().item() * periods_per_year)

        # 主展示年化收益：
        # - 样本 >= 90 天：用 CAGR（可信）
        # - 样本 < 90 天：CAGR 会爆炸（如 10 天 16x → CAGR 9e44），无意义
        #   此时用线性年化但标注"参考值"，并依赖警告说明
        if not is_short_sample:
            ann_ret = cagr
        else:
            # 短样本：CAGR 无意义，用线性值但限制在合理范围（±100000%）
            ann_ret = max(min(ann_ret_linear, 100000.0), -100000.0)
            cagr = cagr  # 保留原始值用于对比展示

        # 最大回撤（基于资金曲线）
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown))

        # 夏普比率（对数收益率空间）
        ret_std = float(bar_logret.std().item())
        raw_sharpe = float(bar_logret.mean().item() / (ret_std + 1e-9) * math.sqrt(periods_per_year))
        sharpe = raw_sharpe
        sharpe_clipped = False  # 夏普未做硬截断，但样本过短时不可信

        # ── 构建警告列表 ──
        warnings = []
        if is_short_sample:
            warnings.append(
                f"样本仅 {data_duration_days:.1f} 天，年化收益/夏普/Sortino 等"
                f"年化指标为外推估计，不代表实际长期表现。"
            )
        if is_in_sample:
            warnings.append(
                f"样本内回测：策略在 {strat_symbol} 上训练，又在同一品种数据上回测，"
                f"绩效必然偏好，请用样本外数据验证。"
            )
        if sortino_clipped:
            warnings.append("Sortino 达到截断上限（±20），原始值过大不可信。")
        if calmar_clipped:
            warnings.append("Calmar 达到截断上限（±10），原始值过大不可信。")
        if sharpe > 5 and is_short_sample:
            warnings.append(
                f"夏普 {sharpe:.1f} 虚高：5 分钟线每年 {periods_per_year:.0f} 根，"
                f"√N 放大因子极大，短样本下意义有限。"
            )

        # 胜率（按 bar 收益正负统计）
        wins = (bar_ret_flat > 0).sum()
        total_trades = (bar_ret_flat != 0).sum()
        win_rate = float(wins / max(total_trades, 1))

        # 换手率
        avg_turnover = float(turnover.mean().item())

        # 在场时间
        exposure = float(position.abs().mean().item())

        # 多空比例
        long_ratio = float((position > 0.05).float().mean().item())
        short_ratio = float((position < -0.05).float().mean().item())

        # 8. 交易明细（采样）
        position_flat = position.reshape(-1).numpy()
        pos_changes = np.diff(position_flat, prepend=0)
        trade_events = []
        for i in range(len(pos_changes)):
            if abs(pos_changes[i]) > 0.05:
                t_idx = i % T
                trade_events.append({
                    "bar": t_idx,
                    "time": time_list[t_idx] if t_idx < len(time_list) else t_idx,
                    "position": round(float(position_flat[i]), 4),
                    "change": round(float(pos_changes[i]), 4),
                })
        # 限制交易明细数量
        if len(trade_events) > 200:
            trade_events = trade_events[:200]

        # 9. 综合评分（bar_logret 即 PnL 对数收益率，传给评估器）
        score = bt._multi_objective(
            factor, target_ret, bar_logret, position, eval_bars=T
        ).item()

        return {
            "strategy": {
                "formula": formula,
                "formula_decoded": formula_decoded,
                "best_score": strategy.get("best_score", 0),
                "symbol": strategy.get("symbol"),
            },
            "data": {
                "file": pathlib.Path(data_file).name,
                "n_bars": T,
                "n_symbols": N,
            },
            "config": {
                "cost_rate": cost_rate,
                "slippage": slippage,
                "initial_capital": initial_capital,
                "leverage": leverage,
            },
            "performance": {
                "total_return": round(total_return * 100, 2),
                "annualized_return": round(ann_ret * 100, 2),
                "annualized_return_cagr": round(max(min(cagr, 1000.0), -1000.0) * 100, 2),
                "annualized_return_linear": round(ann_ret_linear * 100, 2),
                "sortino": round(sortino, 3),
                "sortino_clipped": sortino_clipped,
                "sharpe": round(sharpe, 3),
                "sharpe_clipped": sharpe_clipped,
                "calmar": round(calmar, 3),
                "calmar_clipped": calmar_clipped,
                "max_drawdown": round(max_dd * 100, 2),
                "win_rate": round(win_rate * 100, 1),
                "avg_turnover": round(avg_turnover, 4),
                "exposure": round(exposure * 100, 1),
                "long_ratio": round(long_ratio * 100, 1),
                "short_ratio": round(short_ratio * 100, 1),
                "score": round(score, 3),
            },
            "data_quality": {
                "data_duration_days": round(data_duration_days, 1),
                "data_duration_years": round(data_duration_years, 4),
                "periods_per_year": periods_per_year,
                "is_short_sample": is_short_sample,
                "is_in_sample": is_in_sample,
                "warnings": warnings,
            },
            "equity_curve": {
                "time": time_sampled,
                "equity": [round(e, 2) for e in equity_sampled],
            },
            "trades": trade_events,
            "n_trades": len(trade_events),
        }


# 全局单例
backtest_service = BacktestService()
