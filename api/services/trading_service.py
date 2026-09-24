"""
api/services/trading_service.py — 实盘交易服务

支持 paper / live 双模式，含风控、审计日志。

核心功能：
  - Delta 差额调仓机制（HOLD / CLOSE / REVERSE / ADJUST），彻底杜绝重复加仓
  - ctVal 精确下单（根据合约面值计算张数）
  - 下单附带交易所硬止损单（attachAlgoOrds）防程序离线裸奔
  - 下单前自动 set_leverage
  - 策略安全门禁（实盘过滤负分/退化策略）
  - 单日亏损风控（MAX_DAILY_LOSS_PCT 超限主动清仓）
  - 运行状态显示（账户/持仓/风控/审计统计）
"""
import json
import time
import pathlib
import threading
import os
import logging
from datetime import datetime, timezone
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger("trading_service")

from config import Config
from model.vocab import FORMULA_VOCAB
from model.vm import StackVM
from model.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless, signal_to_action
from data_pipeline.okx_client import OKXClient, get_public_client, get_private_client
from data_pipeline.okx_ws_client import okx_ws_client
from api.services.strategy_service import load_strategy, decode_formula, eval_strategy_factor


class AuditLog:
    """交易审计日志——记录每笔决策。"""

    def __init__(self, log_path: str = "trading_audit.jsonl"):
        self.log_path = pathlib.Path(log_path)
        self._lock = threading.Lock()

    def log(self, event: dict):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event["mode"] = Config.TRADING_MODE
        event["broker_tag"] = Config.OKX_BROKER_TAG
        event["is_live"] = Config.is_live()
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def get_recent(self, n: int = 50) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
        recent = lines[-n:]
        result = []
        for line in recent:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result


class TradingService:
    """交易服务——paper / live 双模式。"""

    def __init__(self):
        self.vm = StackVM()
        self.audit = AuditLog()
        self._position_cache: dict[str, dict] = {}  # inst_id → 当前持仓状态
        self._lock = threading.Lock()
        # 合约信息缓存（ctVal 等），避免每次下单都查询
        self._instrument_cache: dict[str, dict] = {}
        # 账户持仓模式缓存："net_mode" 或 "long_short_mode"，None=未查询
        self._pos_mode_cache: Optional[str] = None
        # 单日盈亏跟踪：{"date": "YYYY-MM-DD", "initial_eq": float|None, "realized_pnl": float}
        self._daily_tracker: dict = {"date": None, "initial_eq": None, "realized_pnl": 0.0}
        # 自动执行调度器状态
        self._auto_trade_state: dict = {"running": False}
        self._auto_trade_thread = None
        # 交易冷却：记录每个品种最后一次下单时间，防止频繁开平
        self._last_order_time: dict[str, float] = {}
        self._cooldown_seconds: int = int(os.getenv("TRADE_COOLDOWN_SECONDS", "180"))
        # 阶梯主动止盈单向棘轮跟踪：inst_id -> 最高生效的仓位上限比例 (0.65 或 0.30)
        self._ladder_tp_ratchet: dict[str, float] = {}
        # 阶梯主动止盈基准持仓跟踪：inst_id -> 触发止盈前的基准总持仓张数
        self._ladder_tp_base_sz: dict[str, float] = {}
        # 高水位价格跟踪：inst_id -> 开仓以来的极值价格（多头为最高价，空头为最低价）
        self._peak_price: dict[str, float] = {}

    # ── 状态查询 ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """获取交易服务状态。"""
        return {
            "mode": Config.TRADING_MODE,
            "is_live": Config.is_live(),
            "is_paper": Config.is_paper(),
            "api_configured": bool(Config.OKX_API_KEY and Config.OKX_API_SECRET and Config.OKX_API_PASSPHRASE),
            "simulated": Config.OKX_API_SIMULATED,
            "active_positions": list(self._position_cache.values()),
            "ws_status": okx_ws_client.get_status(),
            "risk_config": {
                "max_leverage": Config.MAX_LEVERAGE,
                "max_daily_loss_pct": Config.MAX_DAILY_LOSS_PCT,
                "max_position_pct": Config.MAX_POSITION_PCT,
                "trade_cooldown_seconds": self._cooldown_seconds,
            },
        }

    def get_runtime_status(self) -> dict:
        """获取运行时状态：账户、持仓、单日风控、审计统计。

        用于前端「运行状态」面板实时展示。
        """
        status = {
            "mode": Config.TRADING_MODE,
            "is_live": Config.is_live(),
            "ws_status": okx_ws_client.get_status(),
            "account": None,
            "account_error": None,
            "positions": [],
            "positions_error": None,
            "daily_risk": None,
            "position_cache": list(self._position_cache.values()),
            "audit_stats": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 尝试获取账户信息
        try:
            summary = get_private_client().get_account_summary()
            status["account"] = summary
        except Exception as e:
            status["account_error"] = str(e)

        # 尝试获取真实持仓
        try:
            positions = get_private_client().get_positions_detail()
            status["positions"] = positions
        except Exception as e:
            status["positions_error"] = str(e)

        # 交叉对齐兜底：若持仓列表有活跃持仓，而账户汇总因接口差异缺失 upl 或 margin，则从持仓自动补全
        if status["account"] and status["positions"]:
            acct = status["account"]
            pos_list = status["positions"]
            active_pos = [p for p in pos_list if abs(p.get("pos", 0.0)) > 0]
            if active_pos:
                if acct.get("upl", 0.0) == 0.0:
                    pos_upl = sum(p.get("upl", 0.0) for p in active_pos)
                    if pos_upl != 0.0:
                        acct["upl"] = pos_upl
                        total_eq = acct.get("total_eq", 0.0)
                        cost_basis = total_eq - pos_upl
                        acct["upl_ratio"] = (pos_upl / cost_basis) if cost_basis > 0 else (pos_upl / total_eq if total_eq > 0 else 0.0)
                if acct.get("margin", 0.0) == 0.0:
                    pos_margin = sum(p.get("margin", 0.0) for p in active_pos)
                    if pos_margin > 0.0:
                        acct["margin"] = pos_margin
                        total_eq = acct.get("total_eq", 0.0)
                        acct["margin_ratio"] = (pos_margin / total_eq) if total_eq > 0 else 0.0

        # 单日风控状态
        passed, msg, info = self._check_daily_loss()
        status["daily_risk"] = {"passed": passed, "msg": msg, "info": info}

        # 审计统计（今日）
        try:
            recent_logs = self.audit.get_recent(200)
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_logs = [l for l in recent_logs if l.get("timestamp", "").startswith(today_str)]
            status["audit_stats"] = {
                "today_count": len(today_logs),
                "today_executions": sum(1 for l in today_logs if l.get("event") == "signal_execution" and l.get("risk_passed")),
                "today_skips": sum(1 for l in today_logs if l.get("event") == "signal_execution" and not l.get("risk_passed")),
                "today_closes": sum(1 for l in today_logs if l.get("event") == "close_position"),
            }
        except Exception:
            pass

        # 自动交易运行状态
        try:
            status["auto_trade"] = self.get_auto_trade_status()
        except Exception:
            status["auto_trade"] = None

        return status

    def get_audit_log(self, n: int = 50) -> list[dict]:
        """获取最近审计日志。"""
        return self.audit.get_recent(n)

    # ── 风控辅助 ──────────────────────────────────────────────────────────

    def _check_daily_loss(self) -> tuple[bool, str, dict]:
        """检查单日亏损是否超过 MAX_DAILY_LOSS_PCT。

        逻辑：
          - 跨天重置，记录当日初始权益
          - 当前权益低于初始权益 × (1 - MAX_DAILY_LOSS_PCT) 时拒绝下单并触发清仓
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 跨天重置
        if self._daily_tracker["date"] != today:
            self._daily_tracker = {"date": today, "initial_eq": None, "realized_pnl": 0.0}

        # 尝试获取当前账户权益
        current_eq = None
        try:
            summary = get_private_client().get_account_summary()
            if isinstance(summary, dict):
                raw_eq = summary.get("total_eq")
                if raw_eq is not None:
                    try:
                        current_eq = float(raw_eq)
                    except (ValueError, TypeError):
                        current_eq = None
        except Exception:
            current_eq = None

        # 记录当日初始权益（首次获取到时记录）
        if self._daily_tracker["initial_eq"] is None and current_eq and current_eq > 0:
            self._daily_tracker["initial_eq"] = current_eq

        initial_eq = self._daily_tracker["initial_eq"]
        info = {
            "initial_eq": initial_eq,
            "current_eq": current_eq,
            "daily_loss_pct": None,
            "max_daily_loss_pct": Config.MAX_DAILY_LOSS_PCT,
        }

        if initial_eq and current_eq:
            daily_loss_pct = (initial_eq - current_eq) / initial_eq
            daily_loss_pct = max(0.0, daily_loss_pct)  # 只考虑亏损
            info["daily_loss_pct"] = daily_loss_pct
            if daily_loss_pct >= Config.MAX_DAILY_LOSS_PCT:
                return (
                    False,
                    f"当日亏损 {daily_loss_pct:.2%} 超过上限 {Config.MAX_DAILY_LOSS_PCT:.2%}，禁止开仓",
                    info,
                )
            return True, f"当日亏损 {daily_loss_pct:.2%} ≤ {Config.MAX_DAILY_LOSS_PCT:.2%}", info

        # 无法获取账户权益（如 paper 模式无凭证），跳过检查
        return True, "未获取到账户权益，单日亏损检查跳过（paper 模式或无凭证）", info

    # ── 合约信息 ──────────────────────────────────────────────────────────

    def _get_instrument_info(self, inst_id: str) -> dict:
        """获取合约信息（带缓存），用于 ctVal 精确下单。"""
        if inst_id in self._instrument_cache:
            return self._instrument_cache[inst_id]
        try:
            if "SWAP" in inst_id:
                inst_type = "SWAP"
            elif "SPOT" in inst_id:
                inst_type = "SPOT"
            elif "FUTURES" in inst_id:
                inst_type = "FUTURES"
            else:
                inst_type = "SWAP"
            info = get_public_client().get_instrument(inst_id, inst_type)
            if info:
                self._instrument_cache[inst_id] = info
            return info
        except Exception:
            return {}

    def _get_pos_mode(self) -> str:
        """获取账户持仓模式（带缓存）。"""
        if self._pos_mode_cache is not None:
            return self._pos_mode_cache
        try:
            config = get_private_client().get_account_config()
            mode = config.get("posMode", "net_mode")
            self._pos_mode_cache = mode
            return mode
        except Exception:
            self._pos_mode_cache = "net_mode"
            return "net_mode"

    def _compute_target_size(
        self,
        signal: float,
        capital: float,
        max_position_pct: float,
        leverage: int,
        last_price: float,
        inst_info: dict,
    ) -> tuple[float, float, dict]:
        """计算目标仓位（张数），基于 ctVal 精确转换。"""
        max_position_value = capital * max_position_pct
        target_value = signal * max_position_value * leverage

        ct_val = float(inst_info.get("ctVal", 1)) if inst_info else 1.0
        lot_sz = float(inst_info.get("lotSz", 1)) if inst_info else 1.0
        min_sz = float(inst_info.get("minSz", 0)) if inst_info else 0.0
        ct_val_ccy = inst_info.get("ctValCcy", "") if inst_info else ""

        if last_price > 0 and ct_val > 0:
            raw_sz = abs(target_value / (last_price * ct_val))
        else:
            raw_sz = 0.0

        from decimal import Decimal, ROUND_DOWN
        if lot_sz > 0:
            lot_dec = Decimal(str(lot_sz))
            raw_dec = Decimal(str(raw_sz))
            steps = (raw_dec / lot_dec).to_integral_value(rounding=ROUND_DOWN)
            target_sz = float(steps * lot_dec)
        else:
            target_sz = round(raw_sz, 4)

        if lot_sz >= 1:
            sz_str = str(int(target_sz))
        elif lot_sz > 0:
            lot_str = f"{lot_sz:.10f}".rstrip("0").rstrip(".")
            precision = len(lot_str.split(".")[1]) if "." in lot_str else 0
            sz_str = f"{target_sz:.{precision}f}"
        else:
            sz_str = str(round(target_sz, 4))

        below_min = False
        if min_sz > 0 and target_sz < min_sz:
            below_min = True
            target_sz = 0.0

        detail = {
            "ct_val": ct_val,
            "ct_val_ccy": ct_val_ccy,
            "lot_sz": lot_sz,
            "min_sz": min_sz,
            "raw_sz": round(raw_sz, 6),
            "target_sz": round(target_sz, 6),
            "sz_str": sz_str,
            "target_value": round(target_value, 2),
            "below_min": below_min,
            "note": f"1张={ct_val} {ct_val_ccy}, 步进={lot_sz}, 最小={min_sz}",
        }
        return target_sz, target_value, detail

    # ── 持仓状态与 Delta 调仓 ──────────────────────────────────────────────

    def _get_net_position(self, trade_client: OKXClient, inst_id: str) -> tuple[float, list[dict]]:
        """获取当前合约的净持仓张数（多为正，空为负，0为无持仓）及原始持仓列表。"""
        if not Config.is_live():
            with self._lock:
                cached = self._position_cache.get(inst_id)
                if not cached:
                    return 0.0, []
                net_sz = float(cached.get("target_sz", 0.0))
                pos_item = {
                    "inst_id": inst_id,
                    "pos_side": cached.get("pos_side", "long" if net_sz > 0 else "short" if net_sz < 0 else "net"),
                    "pos": abs(net_sz),
                    "avg_px": cached.get("entry_price", cached.get("avg_px", 0.0)),
                    "last": cached.get("entry_price", 0.0),
                    "target_sz": net_sz,
                }
                return net_sz, [pos_item]
        try:
            positions = trade_client.get_positions_detail(inst_id)
        except Exception:
            positions = []
        net_sz = 0.0
        for p in positions:
            sz = float(p.get("pos", 0.0))
            if sz == 0:
                continue
            side = p.get("pos_side", "net")
            if side == "long":
                net_sz += sz
            elif side == "short":
                net_sz -= sz
            elif side == "net":
                net_sz += sz
        return net_sz, positions

    def _handle_position_switch(
        self,
        trade_client: OKXClient,
        inst_id: str,
        new_side: str,
        new_pos_side: str,
    ) -> list[dict]:
        """处理持仓切换：若已有反向持仓，先平仓再开仓。"""
        hedge_actions = []
        try:
            current_positions = trade_client.get_positions_detail(inst_id)
        except Exception:
            current_positions = []

        for pos in current_positions:
            pos_side = pos.get("pos_side", "net")
            pos_sz = pos.get("pos", 0.0)
            if pos_sz == 0:
                continue

            current_is_long = (pos_side == "long" and pos_sz > 0) or (pos_side == "net" and pos_sz > 0)
            current_is_short = (pos_side == "short" and pos_sz > 0) or (pos_side == "net" and pos_sz < 0)

            new_is_long = new_side == "buy"
            new_is_short = new_side == "sell"

            if (current_is_long and new_is_short) or (current_is_short and new_is_long):
                try:
                    close_side = pos_side if pos_side != "net" else "net"
                    close_result = trade_client.close_position(inst_id, pos_side=close_side)
                    trade_client.cancel_algo_orders(inst_id)
                    hedge_actions.append({
                        "action": "close_opposite",
                        "pos_side": pos_side,
                        "pos_sz": pos_sz,
                        "result": close_result,
                    })
                    self.audit.log({
                        "event": "hedge_close",
                        "inst_id": inst_id,
                        "reason": f"反向持仓切换：{pos_side} → {new_pos_side}",
                        "pos_sz": pos_sz,
                        "result": close_result,
                    })
                except Exception as e:
                    hedge_actions.append({
                        "action": "close_opposite",
                        "pos_side": pos_side,
                        "pos_sz": pos_sz,
                        "error": str(e),
                    })

        return hedge_actions

    # ── 核心执行 ──────────────────────────────────────────────────────────

    def execute_signal(
        self,
        strategy_path: str,
        inst_id: str,
        capital: float = 10000.0,
        leverage: int = 5,
        bar: str = "1H",
        max_position_pct: float = 0.30,
        ladder_tp: Optional[bool] = None,
    ) -> dict:
        """执行交易信号——获取最新行情、计算信号、按 Delta 差额调仓。

        安全流程：
          1. 加载策略（实盘门禁：过滤负分/退化策略） + 获取行情
          2. 计算因子 → 仓位信号
          3. 查询合约信息（ctVal、lotSz、minSz）精确计算目标张数
          4. 获取真实持仓，计算调仓差额 Delta
          5. 风控检查（杠杆上限、信号范围、信号阈值、单日亏损、冷却时间）
          6. Delta 状态机：HOLD（维持不重复加仓）、CLOSE（平仓）、REVERSE（反转）、ADJUST（差额增减）
          7. 自动附带交易所硬止损单（attachAlgoOrds）防程序离线裸奔
          8. 审计日志记录全程
        """
        # 风控：杠杆上限
        leverage = min(leverage, Config.MAX_LEVERAGE)

        # 1. 加载策略
        strategy = load_strategy(strategy_path)
        formula = strategy.get("formula")
        formula_decoded = strategy.get("formula_decoded", "")
        best_score = strategy.get("best_score")

        # 策略质量安全门禁：实盘模式下禁止运行评分 <= 0 的策略
        if Config.is_live() and (best_score is None or best_score <= 0):
            risk_checks = [{
                "check": "strategy_quality",
                "passed": False,
                "msg": f"策略评分过低 ({best_score} <= 0)，禁止在实盘模式下执行，请先重新训练",
            }]
            order_result = {
                "skipped": True,
                "reason": f"策略评分过低 ({best_score} <= 0)，禁止在实盘模式下执行",
            }
            audit_event = {
                "event": "signal_execution",
                "inst_id": inst_id,
                "strategy": formula_decoded,
                "bar": bar,
                "signal": 0.0,
                "action": "拒绝执行",
                "risk_checks": risk_checks,
                "risk_passed": False,
                "order": order_result,
            }
            self.audit.log(audit_event)
            return {
                "inst_id": inst_id,
                "bar": bar,
                "last_price": 0.0,
                "signal": 0.0,
                "action": "拒绝执行",
                "target_sz": 0.0,
                "target_value": 0.0,
                "side": "",
                "risk_checks": risk_checks,
                "risk_passed": False,
                "order": order_result,
                "mode": Config.TRADING_MODE,
                "is_live": Config.is_live(),
                "strategy": {
                    "formula": formula,
                    "formula_decoded": formula_decoded,
                    "best_score": best_score,
                },
            }

        # 2. 获取行情（自动分页、升序、只含已收盘 K 线）
        min_bars = getattr(Config, "REALTIME_MIN_BARS", 800)
        client = get_public_client()
        candles = client.get_recent_candles(inst_id, bar, total=min_bars, only_confirmed=True)
        if not candles:
            raise RuntimeError(f"未获取到 {inst_id} 行情")

        close_arr = np.array([float(c[4]) for c in candles], dtype=np.float64)
        open_arr = np.array([float(c[1]) for c in candles], dtype=np.float64)
        high_arr = np.array([float(c[2]) for c in candles], dtype=np.float64)
        low_arr = np.array([float(c[3]) for c in candles], dtype=np.float64)
        vol_arr = np.array([float(c[5]) for c in candles], dtype=np.float64)
        time_arr = np.array([int(c[0]) // 1000 if int(c[0]) > 1e12 else int(c[0]) for c in candles], dtype=np.float64)

        if len(candles) < min_bars:
            raise RuntimeError(
                f"历史 bar 不足（{len(candles)}/{min_bars}），"
                f"无法稳定计算特征与滚动归一化，拒绝出信号"
            )

        raw_dict = {
            "close": torch.from_numpy(close_arr).unsqueeze(0).float(),
            "open": torch.from_numpy(open_arr).unsqueeze(0).float(),
            "high": torch.from_numpy(high_arr).unsqueeze(0).float(),
            "low": torch.from_numpy(low_arr).unsqueeze(0).float(),
            "volume": torch.from_numpy(vol_arr).unsqueeze(0).float(),
            "time": time_arr,
        }

        # 3. 计算信号（兼容单策略与多因子组合）
        feat = MT5FeatureEngineer.compute_features(raw_dict)
        with torch.no_grad():
            factor = eval_strategy_factor(strategy, self.vm, feat)
        if factor is None:
            raise ValueError("因子计算失败（无效策略或求值异常）")

        # 3.1 Reef Adaptive Harness: 探测微观体制并动态获取风控/Neutral Band 参数
        harness_params = None
        lower_band = 0.25
        upper_band = 0.75
        dynamic_sl_pct = getattr(Config, "HARNESS_DEFAULT_SL", 0.030)
        try:
            if getattr(Config, "ENABLE_DYNAMIC_HARNESS", True):
                from evolution.harness import get_harness_policy
                harness_params = get_harness_policy().evaluate(raw_dict)
                lower_band = harness_params.lower_band
                upper_band = harness_params.upper_band
                dynamic_sl_pct = harness_params.stop_loss_pct
        except Exception as e:
            logger.warning(f"Reef Adaptive Harness 评估异常，回退默认风控: {e}")

        position_signal = compute_target_positions_stateless(
            factor, lower_band=lower_band, upper_band=upper_band
        )
        signal = float(position_signal[0, -1].item())
        last_price = float(close_arr[-1])

        # 3.5 因子诊断信息
        factor_val = float(factor[0, -1].item()) if hasattr(factor, 'dim') and callable(factor.dim) and factor.dim() >= 2 else float(factor[-1].item())
        tanh_val = float(torch.tanh(torch.tensor(factor_val)).item())
        signal_diag = {
            "factor": round(factor_val, 6),
            "tanh": round(tanh_val, 6),
            "abs_tanh": round(abs(tanh_val), 6),
            "lower_band": lower_band,
            "upper_band": upper_band,
            "in_neutral_band": abs(tanh_val) < lower_band,
            "raw_signal_before_band": round(tanh_val, 6),
            "signal_after_band": round(signal, 6),
            "bars_used": len(close_arr),
            "feat_shape": list(feat.shape) if hasattr(feat, 'shape') else None,
            "harness": harness_params.to_dict() if harness_params else None,
        }

        # 3.6 参数一致性与因子方差健康度检查
        param_warnings = []
        recent_std = float(factor[0, -min(200, factor.shape[1]):].std().item()) if factor.numel() > 0 else 0.0
        signal_diag["recent_std"] = round(recent_std, 6)
        if recent_std < 1e-4:
            param_warnings.append(f"策略因子在最近周期几乎无波动 (std={recent_std:.6f} < 1e-4)，策略已进入常数退化状态")
        if bar != "1H":
            param_warnings.append(f"K线周期 {bar} ≠ 训练周期 1H（信号分布可能不同）")
        if leverage != Config.DEFAULT_LEVERAGE:
            param_warnings.append(f"杠杆 {leverage}x ≠ 默认 {Config.DEFAULT_LEVERAGE}x")
        if capital < 100:
            param_warnings.append(f"本金 {capital} USDT 过低（手续费占比过高）")
        if max_position_pct > 0.50:
            param_warnings.append(f"仓位占比 {max_position_pct:.0%} 过高（单笔风险过大）")

        # 4. 查询合约信息 + 计算目标仓位张数与 Delta 差额
        inst_info = self._get_instrument_info(inst_id)
        raw_target_sz, target_value, size_detail = self._compute_target_size(
            signal, capital, max_position_pct, leverage, last_price, inst_info
        )

        lot_sz = float(inst_info.get("lotSz", 1.0)) if inst_info else 1.0
        min_sz = float(inst_info.get("minSz", 0.0)) if inst_info else 0.0
        ct_val = float(inst_info.get("ctVal", 1.0)) if inst_info else float(size_detail.get("ct_val", 1.0))
        step_sz = max(lot_sz, min_sz, 1e-6)

        # 计算目标净持仓（多为正，空为负，平仓/低于最小量为0）
        if abs(signal) < 0.05 or raw_target_sz <= 0 or size_detail.get("below_min"):
            target_held_sz = 0.0
        elif signal > 0:
            target_held_sz = raw_target_sz
        else:
            target_held_sz = -raw_target_sz

        trade_client = get_private_client()
        net_current_sz, current_positions = self._get_net_position(trade_client, inst_id)

        # 4.4 阶梯主动止盈 (Ladder Take Profit)
        use_ladder_tp = Config.ENABLE_LADDER_TP if ladder_tp is None else bool(ladder_tp)
        ladder_tp_diag = {
            "enabled": use_ladder_tp,
            "triggered": False,
            "tier": 0,
            "pnl_pct": 0.0,
            "cap_ratio": 1.0,
            "original_target_sz": round(target_held_sz, 6),
            "adjusted_target_sz": round(target_held_sz, 6),
        }

        # 仓位归零或反向时，重置单向棘轮、基准持仓与高水位极值价格
        if abs(net_current_sz) == 0 or (target_held_sz * net_current_sz < 0):
            with self._lock:
                self._ladder_tp_ratchet.pop(inst_id, None)
                self._ladder_tp_base_sz.pop(inst_id, None)
                self._peak_price.pop(inst_id, None)
        elif target_held_sz != 0 and self._ladder_tp_ratchet.get(inst_id, 1.0) == 1.0:
            with self._lock:
                self._ladder_tp_base_sz[inst_id] = max(abs(net_current_sz), abs(target_held_sz))
                if inst_id not in self._peak_price and last_price > 0:
                    self._peak_price[inst_id] = last_price

        if use_ladder_tp and net_current_sz != 0 and (target_held_sz * net_current_sz > 0):
            # 已有持仓且信号同向（顺势持仓中）：检测纯标的价差浮盈
            active_pos = [p for p in current_positions if float(p.get("pos", 0.0) or p.get("target_sz", 0.0) or 0.0) != 0]
            entry_px = None
            if active_pos:
                p_entry = active_pos[0]
                entry_px = float(p_entry.get("avg_px") or p_entry.get("entry_price") or p_entry.get("avgPx") or 0.0)

            if entry_px and entry_px > 0 and last_price > 0:
                if net_current_sz > 0:  # 多头
                    pnl_pct = (last_price - entry_px) / entry_px
                else:  # 空头
                    pnl_pct = (entry_px - last_price) / entry_px

                ladder_tp_diag["pnl_pct"] = round(pnl_pct, 6)

                tier = 0
                cap_ratio = 1.0
                if pnl_pct >= Config.LADDER_TP_TIER2_PCT:
                    tier = 2
                    cap_ratio = Config.LADDER_TP_TIER2_CAP
                elif pnl_pct >= Config.LADDER_TP_TIER1_PCT:
                    tier = 1
                    cap_ratio = Config.LADDER_TP_TIER1_CAP

                # 单向棘轮：锁定历史达到的最高阶梯（只允许仓位上限变紧，绝不放宽回补）
                with self._lock:
                    prev_ratchet = self._ladder_tp_ratchet.get(inst_id, 1.0)
                    effective_cap = min(prev_ratchet, cap_ratio)
                    if tier > 0:
                        self._ladder_tp_ratchet[inst_id] = effective_cap

                if effective_cap < 1.0:
                    ladder_tp_diag["triggered"] = True
                    ladder_tp_diag["tier"] = 2 if effective_cap <= Config.LADDER_TP_TIER2_CAP else 1
                    ladder_tp_diag["cap_ratio"] = effective_cap

                    # 目标张数压缩：不得高于 base_sz * effective_cap
                    base_sz = self._ladder_tp_base_sz.get(inst_id, abs(target_held_sz))
                    max_allowed_sz = base_sz * effective_cap
                    # 单向棘轮约束：目标张数不得大于当前净持仓（防止价格回调时反向回补追买）
                    capped_abs_sz = min(abs(net_current_sz), max_allowed_sz)

                    from decimal import Decimal, ROUND_DOWN
                    if lot_sz > 0:
                        lot_dec = Decimal(str(lot_sz))
                        capped_dec = Decimal(str(capped_abs_sz))
                        steps = (capped_dec / lot_dec).to_integral_value(rounding=ROUND_DOWN)
                        capped_sz = float(steps * lot_dec)
                    else:
                        capped_sz = round(capped_abs_sz, 4)

                    if capped_sz < min_sz:
                        capped_sz = 0.0

                    target_held_sz = capped_sz if target_held_sz > 0 else -capped_sz
                    ladder_tp_diag["adjusted_target_sz"] = round(target_held_sz, 6)

        delta_sz = target_held_sz - net_current_sz

        pos_mode = self._get_pos_mode()

        # 4.5 获取账户余额（用于保证金检查）
        account_eq = None
        account_avail = None
        try:
            acct = trade_client.get_account_summary()
            account_eq = acct.get("total_eq")
            account_avail = acct.get("avail_bal")
        except Exception:
            pass

        # 5. 风控检查
        risk_checks = []
        risk_passed = True

        # 5.1 杠杆上限
        if leverage > Config.MAX_LEVERAGE:
            risk_checks.append({"check": "leverage", "passed": False, "msg": f"杠杆 {leverage} 超过上限 {Config.MAX_LEVERAGE}"})
            risk_passed = False
        else:
            risk_checks.append({"check": "leverage", "passed": True, "msg": f"杠杆 {leverage} ≤ {Config.MAX_LEVERAGE}"})

        # 5.2 信号范围
        if abs(signal) > 1.0:
            risk_checks.append({"check": "signal_range", "passed": False, "msg": f"信号 {signal:.2f} 超出 [-1,1]"})
            risk_passed = False
        else:
            risk_checks.append({"check": "signal_range", "passed": True, "msg": f"信号 {signal:.2f} 在范围内"})

        # 5.3 交易冷却检查（仅在需要调仓开仓/加仓/反转时检查，平仓与减仓保护性退出不受冷却限制）
        is_reduction_or_close = (target_held_sz == 0.0 and abs(net_current_sz) > 0) or (
            target_held_sz * net_current_sz > 0 and abs(target_held_sz) < abs(net_current_sz)
        )
        needs_order = not (
            (target_held_sz == 0.0 and abs(net_current_sz) == 0.0)
            or (abs(delta_sz) < step_sz and target_held_sz * net_current_sz > 0)
            or is_reduction_or_close
        )
        if risk_passed and needs_order:
            last_t = self._last_order_time.get(inst_id, 0)
            elapsed = time.time() - last_t
            if elapsed < self._cooldown_seconds:
                risk_checks.append({
                    "check": "trade_cooldown",
                    "passed": False,
                    "msg": f"冷却中：距上次下单 {elapsed:.0f}s < {self._cooldown_seconds}s",
                })
                risk_passed = False
            else:
                risk_checks.append({
                    "check": "trade_cooldown",
                    "passed": True,
                    "msg": f"冷却已过：距上次下单 {elapsed:.0f}s ≥ {self._cooldown_seconds}s",
                })

        # 5.4 单日亏损风控（若超限则拒绝开仓，并主动清仓保护）
        daily_passed, daily_msg, daily_info = self._check_daily_loss()
        risk_checks.append({"check": "daily_loss", "passed": daily_passed, "msg": daily_msg, "info": daily_info})
        if not daily_passed:
            risk_passed = False
            if abs(net_current_sz) > 0:
                try:
                    close_res = self.close_position(inst_id)
                    self.audit.log({
                        "event": "daily_risk_liquidation",
                        "inst_id": inst_id,
                        "reason": f"单日亏损超限触发清仓: {daily_msg}",
                        "close_result": close_res,
                    })
                except Exception as e:
                    self.audit.log({
                        "event": "daily_risk_liquidation_error",
                        "inst_id": inst_id,
                        "error": str(e),
                    })

        # 5.5 保证金充足性检查
        # 规则说明：
        # a) 减仓/平仓（同向减持或完全清仓）只会释放保证金并落袋盈亏，绝不消耗可用余额，无条件放行；
        # b) 同向加仓仅校验增量 delta_sz 所需的新增保证金；
        # c) 空仓新开校验目标名义仓位所需保证金；反向开仓会先全平旧仓位释放保证金，结合账户总权益评估。
        if risk_passed and account_avail is not None and abs(delta_sz) > 0:
            is_reduction = (target_held_sz * net_current_sz > 0 and abs(target_held_sz) <= abs(net_current_sz)) or (target_held_sz == 0 and net_current_sz != 0)
            if is_reduction:
                risk_checks.append({
                    "check": "margin_sufficiency",
                    "passed": True,
                    "msg": f"减仓/止盈操作释放保证金，当前可用余额 {account_avail:.2f} USDT 充足",
                })
            else:
                if target_held_sz * net_current_sz > 0 and abs(target_held_sz) > abs(net_current_sz):
                    # 同向增仓，只需校验新增增量部分的保证金
                    incremental_value = abs(delta_sz) * ct_val * last_price
                    required_margin = incremental_value / leverage if leverage > 0 else incremental_value
                else:
                    # 空仓新开或反向开仓
                    required_margin = abs(target_value) / leverage if leverage > 0 else abs(target_value)

                # 反向开仓会先平旧仓释放保证金，因而可用资金有效容量应计入释放后的权益
                effective_avail = account_avail
                if target_held_sz * net_current_sz < 0 and account_eq is not None:
                    effective_avail = max(account_avail, account_eq)

                if effective_avail < required_margin:
                    risk_checks.append({
                        "check": "margin_sufficiency",
                        "passed": False,
                        "msg": f"可用余额 {effective_avail:.2f} USDT < 所需保证金 {required_margin:.2f} USDT",
                    })
                    risk_passed = False
                else:
                    risk_checks.append({
                        "check": "margin_sufficiency",
                        "passed": True,
                        "msg": f"可用余额 {effective_avail:.2f} USDT ≥ 所需保证金 {required_margin:.2f} USDT",
                    })

        # 6. Delta 状态机执行调仓
        order_result = None
        hedge_actions = []
        leverage_result = None
        executed_side = ""

        if not risk_passed:
            order_result = {
                "skipped": True,
                "reason": "风控未通过",
                "signal": round(signal, 4),
            }
        elif target_held_sz == 0.0 and abs(net_current_sz) > 0:
            # 状态 1: 平仓 (CLOSE)
            close_res = self.close_position(inst_id)
            order_result = {
                "action": "CLOSE",
                "reason": f"目标持仓为 0，平掉当前持仓 ({net_current_sz:.4f})",
                "close_result": close_res,
                "live": Config.is_live(),
            }
            with self._lock:
                self._position_cache.pop(inst_id, None)
                self._last_order_time[inst_id] = time.time()
        elif target_held_sz == 0.0 and abs(net_current_sz) == 0:
            # 状态 2: 保持空仓 (FLAT)
            order_result = {
                "action": "HOLD",
                "skipped": True,
                "reason": "当前无持仓且目标为空仓",
                "live": False,
            }
        elif abs(delta_sz) < step_sz and (target_held_sz * net_current_sz > 0):
            # 状态 3: 维持现有持仓 (HOLD) —— 关键修复：同向不重复开仓！
            order_result = {
                "action": "HOLD",
                "skipped": True,
                "reason": f"持仓 ({net_current_sz:.4f}) 已达到目标 ({target_held_sz:.4f})，维持现有持仓",
                "live": False,
            }
        else:
            # 状态 4: 发送调仓订单（开新仓、反转、或增减仓）
            # 4.1 设置杠杆
            try:
                lev_pos_side = "" if pos_mode == "net_mode" else ("long" if target_held_sz > 0 else "short")
                leverage_result = trade_client.set_leverage(
                    inst_id, lever=leverage, mgn_mode="cross",
                    pos_side=lev_pos_side,
                )
            except Exception as e:
                leverage_result = {"error": str(e)}

            # 4.2 计算具体调仓量与方向
            # 若反转持仓 (由多翻空或由空翻多)：先全平旧仓位，再按目标全量开新仓
            if net_current_sz * target_held_sz < 0:
                try:
                    close_res = self.close_position(inst_id)
                    hedge_actions.append({"action": "reverse_close", "result": close_res})
                except Exception as e:
                    logger.warning(f"反转平旧仓异常: {e}")
                    hedge_actions.append({"action": "reverse_close_error", "error": str(e)})
                order_sz = abs(target_held_sz)
                order_side = "buy" if target_held_sz > 0 else "sell"
                order_pos_side = "long" if target_held_sz > 0 else "short"
            elif net_current_sz == 0:
                order_sz = abs(target_held_sz)
                order_side = "buy" if target_held_sz > 0 else "sell"
                order_pos_side = "long" if target_held_sz > 0 else "short"
            else:
                # 同向增减仓
                order_sz = abs(delta_sz)
                order_side = "buy" if delta_sz > 0 else "sell"
                order_pos_side = "long" if net_current_sz > 0 else "short"

            executed_side = order_side
            if pos_mode == "net_mode":
                order_pos_side = "net"

            # 4.3 交易所硬止损价（基于 Reef Adaptive Harness 动态止损或默认 3% 止损保护）
            # 减仓时（如阶梯止盈）不附带反向硬止损，仅在开新仓、增仓或反转时附带硬止损
            is_reduction = (target_held_sz * net_current_sz > 0) and (abs(target_held_sz) < abs(net_current_sz))
            if is_reduction:
                sl_price_str = None
            else:
                stop_loss_pct = dynamic_sl_pct if harness_params else 0.03
                if order_side == "buy":
                    sl_price = last_price * (1.0 - stop_loss_pct)
                else:
                    sl_price = last_price * (1.0 + stop_loss_pct)
                tick_sz = float(inst_info.get("tickSz", 0.01)) if inst_info else 0.01
                tick_dec = len(str(tick_sz).split(".")[1]) if "." in str(tick_sz) else 2
                sl_price_str = f"{sl_price:.{tick_dec}f}"

            # 4.4 格式化下单数量 sz
            if lot_sz >= 1:
                sz_str = str(int(round(order_sz)))
            elif lot_sz > 0:
                lot_str = f"{lot_sz:.10f}".rstrip("0").rstrip(".")
                lot_precision = len(lot_str.split(".")[1]) if "." in lot_str else 0
                sz_str = f"{order_sz:.{lot_precision}f}"
            else:
                sz_str = str(round(order_sz, 4))

            if order_sz < min_sz or order_sz <= 0:
                order_result = {
                    "skipped": True,
                    "reason": f"调仓量 {sz_str} 低于最小下单量 minSz={min_sz}",
                    "live": False,
                }
            else:
                inst_clean = ''.join(c for c in inst_id if c.isalnum())[:8]
                cl_ord_id = f"ap{int(time.time() * 1000)}{inst_clean}"[:32]

                # 4.5 生成 Reef Observe 决策收据
                receipt_id = None
                try:
                    from evolution.observer import get_observer
                    receipt_id = get_observer().record_receipt(
                        inst_id=inst_id,
                        side=order_side,
                        pos_side=order_pos_side,
                        signal=signal,
                        action=signal_to_action(signal),
                        last_price=last_price,
                        target_sz=target_held_sz,
                        delta_sz=delta_sz,
                        strategy_formula=formula_decoded,
                        bar=bar,
                        cl_ord_id=cl_ord_id,
                        extra={
                            "size_detail": size_detail,
                            "best_score": best_score,
                            "harness": harness_params.to_dict() if harness_params else None,
                        },
                    )
                except Exception:
                    pass

                order_result = trade_client.place_order(
                    inst_id=inst_id,
                    side=order_side,
                    pos_side=order_pos_side,
                    ord_type="market",
                    sz=sz_str,
                    td_mode="cross",
                    cl_ord_id=cl_ord_id,
                    sl_trigger_px=sl_price_str,
                )

                # 更新持仓缓存
                with self._lock:
                    self._position_cache[inst_id] = {
                        "inst_id": inst_id,
                        "pos_side": order_pos_side,
                        "pos": abs(round(target_held_sz, 6)),
                        "side": order_side,
                        "signal": round(signal, 4),
                        "target_sz": round(target_held_sz, 6),
                        "target_value": round(target_value, 2),
                        "avg_px": last_price,
                        "entry_price": last_price,
                        "stop_loss_price": sl_price_str,
                        "capital": capital,
                        "leverage": leverage,
                        "time": int(time.time()),
                        "simulated": not Config.is_live(),
                        "size_detail": size_detail,
                        "receipt_id": receipt_id,
                    }
                    self._last_order_time[inst_id] = time.time()

        # 7. 审计日志
        receipt_id_val = locals().get("receipt_id")
        audit_event = {
            "event": "signal_execution",
            "inst_id": inst_id,
            "strategy": formula_decoded,
            "bar": bar,
            "last_price": last_price,
            "signal": round(signal, 4),
            "action": signal_to_action(signal),
            "current_held_sz": round(net_current_sz, 6),
            "target_held_sz": round(target_held_sz, 6),
            "delta_sz": round(delta_sz, 6),
            "capital": capital,
            "leverage": leverage,
            "max_position_pct": max_position_pct,
            "size_detail": size_detail,
            "risk_checks": risk_checks,
            "risk_passed": risk_passed,
            "hedge_actions": hedge_actions,
            "leverage_result": leverage_result,
            "order": order_result,
            "receipt_id": receipt_id_val,
            "harness": harness_params.to_dict() if harness_params else None,
            "param_warnings": param_warnings if param_warnings else None,
            "ladder_tp": ladder_tp_diag,
        }
        self.audit.log(audit_event)

        return {
            "inst_id": inst_id,
            "bar": bar,
            "last_price": last_price,
            "signal": round(signal, 4),
            "action": signal_to_action(signal),
            "current_held_sz": round(net_current_sz, 6),
            "target_held_sz": round(target_held_sz, 6),
            "delta_sz": round(delta_sz, 6),
            "side": executed_side,
            "size_detail": size_detail,
            "signal_diag": signal_diag,
            "ladder_tp": ladder_tp_diag,
            "harness": harness_params.to_dict() if harness_params else None,
            "risk_checks": risk_checks,
            "risk_passed": risk_passed,
            "hedge_actions": hedge_actions,
            "leverage_result": leverage_result,
            "order": order_result,
            "receipt_id": receipt_id_val,
            "param_warnings": param_warnings if param_warnings else None,
            "mode": Config.TRADING_MODE,
            "is_live": Config.is_live(),
            "strategy": {
                "formula": formula,
                "formula_decoded": formula_decoded,
                "best_score": best_score,
            },
        }

    def close_position(self, inst_id: str) -> dict:
        """平仓并撤销止损单。"""
        trade_client = get_private_client()

        pos_side_to_close = "net"
        if not Config.is_live():
            with self._lock:
                cached = self._position_cache.get(inst_id)
                if cached:
                    pos_side_to_close = cached.get("pos_side", "net")
        try:
            positions = trade_client.get_positions_detail(inst_id)
            if positions:
                for p in positions:
                    if p.get("pos", 0) != 0:
                        pos_side_to_close = p.get("pos_side", "net")
                        break
        except Exception:
            pass

        result = trade_client.close_position(inst_id, pos_side=pos_side_to_close)
        trade_client.cancel_algo_orders(inst_id)

        with self._lock:
            self._position_cache.pop(inst_id, None)
            self._ladder_tp_ratchet.pop(inst_id, None)
            self._ladder_tp_base_sz.pop(inst_id, None)
            self._peak_price.pop(inst_id, None)

        self.audit.log({
            "event": "close_position",
            "inst_id": inst_id,
            "pos_side": pos_side_to_close,
            "result": result,
        })

        # 触发 Reef Observe 异步对齐
        try:
            from evolution.observer import get_observer
            get_observer().sync(client=trade_client, audit_log_path=str(self.audit.log_path))
        except Exception:
            pass

        return {
            "inst_id": inst_id,
            "pos_side": pos_side_to_close,
            "result": result,
            "mode": Config.TRADING_MODE,
            "is_live": Config.is_live(),
        }


    # ── 自动执行调度器 ────────────────────────────────────────────────────

    def check_realtime_ladder_tp(self, inst_id: str) -> Optional[dict]:
        """盘中实时阶梯主动止盈监听（5秒级独立轮询）。

        解耦策略周期的信号计算（如1H），在持仓期间实时监控标的最新价。
        当纯价格浮盈达到阶梯阈值时，立即向交易所报单减仓止盈，并通过单向棘轮锁定利润。
        """
        if not getattr(Config, "ENABLE_LADDER_TP", True):
            return None

        trade_client = get_private_client()
        try:
            net_current_sz, current_positions = self._get_net_position(trade_client, inst_id)
        except Exception:
            return None

        # 仓位为0：重置单向棘轮与基准持仓
        if abs(net_current_sz) == 0:
            with self._lock:
                self._ladder_tp_ratchet.pop(inst_id, None)
                self._ladder_tp_base_sz.pop(inst_id, None)
            return None

        # 获取开仓持仓均价
        entry_px = None
        active_pos = [p for p in current_positions if float(p.get("pos", 0.0) or p.get("target_sz", 0.0) or 0.0) != 0]
        if active_pos:
            p_entry = active_pos[0]
            entry_px = float(p_entry.get("avg_px") or p_entry.get("entry_price") or p_entry.get("avgPx") or 0.0)
        if not entry_px or entry_px <= 0:
            with self._lock:
                cached = self._position_cache.get(inst_id)
                if cached:
                    entry_px = float(cached.get("entry_price") or 0.0)

        if not entry_px or entry_px <= 0:
            return None

        # 获取标的最新价（优先 WebSocket 实时推送信道，无延迟且不消耗 REST 配额；降级 REST ticker）
        last_price = 0.0
        ws_ticker = okx_ws_client.latest_tickers.get(inst_id)
        if ws_ticker and ws_ticker.get("last", 0.0) > 0:
            last_price = float(ws_ticker["last"])
        else:
            try:
                ticker = get_public_client().get_ticker(inst_id)
                last_price = float(ticker.get("last", 0.0) or 0.0)
            except Exception:
                pass

        if last_price <= 0:
            return None

        # 计算纯标的价差浮盈百分比（不受杠杆与保证金放大影响）
        if net_current_sz > 0:  # 多头
            pnl_pct = (last_price - entry_px) / entry_px
        else:  # 空头
            pnl_pct = (entry_px - last_price) / entry_px

        # 动态维护极值价格（多头最高价，空头最低价）
        with self._lock:
            if net_current_sz > 0:
                self._peak_price[inst_id] = max(self._peak_price.get(inst_id, last_price), last_price)
            else:
                self._peak_price[inst_id] = min(self._peak_price.get(inst_id, last_price), last_price)
            peak_px = self._peak_price[inst_id]
            prev_ratchet = self._ladder_tp_ratchet.get(inst_id, 1.0)

        # ── 1. 高水位动态追踪止损 (High-Water Trailing Stop) ─────────────────
        # 条件：开启追踪止损 且 阶梯止盈已达到 Tier 2 (仓位已被限至 30% 趋势底仓)
        if getattr(Config, "ENABLE_TRAILING_SL", True) and prev_ratchet <= Config.LADDER_TP_TIER2_CAP:
            trailing_cb = getattr(Config, "TRAILING_STOP_CALLBACK_PCT", 0.025)
            trailing_triggered = False
            trailing_sl_px = 0.0

            if net_current_sz > 0:  # 多头
                # 追踪止损线 = max(峰值回撤2.5%, 锁定Tier 1利润线)
                trailing_sl_px = max(peak_px * (1.0 - trailing_cb), entry_px * (1.0 + Config.LADDER_TP_TIER1_PCT))
                if last_price <= trailing_sl_px:
                    trailing_triggered = True
            else:  # 空头
                trailing_sl_px = min(peak_px * (1.0 + trailing_cb), entry_px * (1.0 - Config.LADDER_TP_TIER1_PCT))
                if last_price >= trailing_sl_px:
                    trailing_triggered = True

            if trailing_triggered:
                # 触发追踪止损：全平剩余趋势底仓并撤销所有挂单
                close_res = self.close_position(inst_id)
                with self._lock:
                    if self._auto_trade_state.get("running"):
                        self._auto_trade_state["total_orders"] = self._auto_trade_state.get("total_orders", 0) + 1
                        self._auto_trade_state["last_exit"] = {
                            "type": "trailing_sl",
                            "time": time.time(),
                            "exit_price": last_price,
                            "entry_px": entry_px,
                            "peak_price": peak_px,
                            "trailing_sl_px": round(trailing_sl_px, 4),
                            "pnl_pct": round(pnl_pct, 6),
                        }

                audit_data = {
                    "event": "trailing_stop_loss_trigger",
                    "inst_id": inst_id,
                    "entry_px": entry_px,
                    "peak_price": peak_px,
                    "trailing_sl_px": round(trailing_sl_px, 4),
                    "exit_price": last_price,
                    "pnl_pct": round(pnl_pct, 6),
                    "held_sz": round(net_current_sz, 6),
                    "result": close_res,
                }
                self.audit.log(audit_data)
                return audit_data

        # ── 2. 保本止损联动 (Breakeven Stop Loss) ────────────────────────────
        # 条件：开启保本止损 且 阶梯止盈已达到 Tier 1 (仓位已被限至 65%)
        if getattr(Config, "ENABLE_BREAKEVEN_SL", True) and prev_ratchet <= Config.LADDER_TP_TIER1_CAP:
            be_buffer = getattr(Config, "BREAKEVEN_BUFFER_PCT", 0.0015)
            be_triggered = False
            be_sl_px = 0.0

            if net_current_sz > 0:  # 多头
                be_sl_px = entry_px * (1.0 + be_buffer)
                if last_price <= be_sl_px:
                    be_triggered = True
            else:  # 空头
                be_sl_px = entry_px * (1.0 - be_buffer)
                if last_price >= be_sl_px:
                    be_triggered = True

            if be_triggered:
                # 触发保本止损：全平剩余仓位并撤销挂单，彻底消灭亏损
                close_res = self.close_position(inst_id)
                with self._lock:
                    if self._auto_trade_state.get("running"):
                        self._auto_trade_state["total_orders"] = self._auto_trade_state.get("total_orders", 0) + 1
                        self._auto_trade_state["last_exit"] = {
                            "type": "breakeven_sl",
                            "time": time.time(),
                            "exit_price": last_price,
                            "entry_px": entry_px,
                            "breakeven_price": round(be_sl_px, 4),
                            "pnl_pct": round(pnl_pct, 6),
                        }

                audit_data = {
                    "event": "breakeven_stop_loss_trigger",
                    "inst_id": inst_id,
                    "entry_px": entry_px,
                    "breakeven_price": round(be_sl_px, 4),
                    "exit_price": last_price,
                    "pnl_pct": round(pnl_pct, 6),
                    "held_sz": round(net_current_sz, 6),
                    "result": close_res,
                }
                self.audit.log(audit_data)
                return audit_data

        # ── 3. 匹配阶梯主动止盈 ──────────────────────────────────────────────
        tier = 0
        cap_ratio = 1.0
        if pnl_pct >= Config.LADDER_TP_TIER2_PCT:
            tier = 2
            cap_ratio = Config.LADDER_TP_TIER2_CAP
        elif pnl_pct >= Config.LADDER_TP_TIER1_PCT:
            tier = 1
            cap_ratio = Config.LADDER_TP_TIER1_CAP

        if tier == 0:
            return None

        with self._lock:
            prev_ratchet = self._ladder_tp_ratchet.get(inst_id, 1.0)
            if inst_id not in self._ladder_tp_base_sz or self._ladder_tp_base_sz[inst_id] <= 0:
                self._ladder_tp_base_sz[inst_id] = abs(net_current_sz)
            base_sz = self._ladder_tp_base_sz[inst_id]

        # 单向棘轮：只允许更严格的止盈档位生效（cap_ratio 必须严格小于 prev_ratchet）
        if cap_ratio >= prev_ratchet:
            return None

        # 计算止盈后保留的目标张数
        max_allowed_sz = base_sz * cap_ratio
        capped_abs_sz = min(abs(net_current_sz), max_allowed_sz)

        inst_info = self._get_instrument_info(inst_id)
        lot_sz = float(inst_info.get("lotSz", 1.0)) if inst_info else 1.0
        min_sz = float(inst_info.get("minSz", 0.0)) if inst_info else 0.0

        from decimal import Decimal, ROUND_DOWN
        if lot_sz > 0:
            lot_dec = Decimal(str(lot_sz))
            capped_dec = Decimal(str(capped_abs_sz))
            steps = (capped_dec / lot_dec).to_integral_value(rounding=ROUND_DOWN)
            target_abs = float(steps * lot_dec)
        else:
            target_abs = round(capped_abs_sz, 4)

        if target_abs < min_sz:
            target_abs = 0.0

        target_held_sz = target_abs if net_current_sz > 0 else -target_abs
        delta_sz = target_held_sz - net_current_sz
        order_sz = abs(delta_sz)

        # 锁定单向棘轮，避免重复触发
        with self._lock:
            self._ladder_tp_ratchet[inst_id] = cap_ratio

        if order_sz < min_sz or order_sz <= 0:
            return None

        # 减仓方向与 pos_side
        order_side = "sell" if net_current_sz > 0 else "buy"
        pos_mode = self._get_pos_mode()
        order_pos_side = "long" if net_current_sz > 0 else "short"
        if pos_mode == "net_mode":
            order_pos_side = "net"

        # 格式化数量
        if lot_sz >= 1:
            sz_str = str(int(order_sz))
        elif lot_sz > 0:
            lot_precision = len(f"{lot_sz:.10f}".rstrip("0").split(".")[1]) if "." in f"{lot_sz:.10f}".rstrip("0") else 0
            sz_str = f"{order_sz:.{lot_precision}f}"
        else:
            sz_str = str(round(order_sz, 4))

        inst_clean = ''.join(c for c in inst_id if c.isalnum())[:8]
        cl_ord_id = f"aptp{int(time.time())}{inst_clean}"

        # 下单减仓止盈（减仓不附带硬止损单）
        try:
            order_result = trade_client.place_order(
                inst_id=inst_id,
                side=order_side,
                pos_side=order_pos_side,
                ord_type="market",
                sz=sz_str,
                td_mode="cross",
                cl_ord_id=cl_ord_id,
            )
        except Exception as e:
            order_result = {"error": str(e), "failed": True}

        with self._lock:
            if not Config.is_live() and inst_id in self._position_cache:
                self._position_cache[inst_id]["target_sz"] = round(target_held_sz, 6)
            self._last_order_time[inst_id] = time.time()
            if self._auto_trade_state.get("running"):
                self._auto_trade_state["total_orders"] = self._auto_trade_state.get("total_orders", 0) + 1
                self._auto_trade_state["last_ladder_tp"] = {
                    "time": time.time(),
                    "tier": tier,
                    "cap_ratio": cap_ratio,
                    "pnl_pct": round(pnl_pct, 6),
                    "target_sz": round(target_held_sz, 6),
                    "order_sz": round(order_sz, 6),
                }

        audit_data = {
            "event": "realtime_ladder_tp_trigger",
            "inst_id": inst_id,
            "tier": tier,
            "cap_ratio": cap_ratio,
            "pnl_pct": round(pnl_pct, 6),
            "entry_px": entry_px,
            "last_price": last_price,
            "base_sz": base_sz,
            "net_current_sz": round(net_current_sz, 6),
            "target_held_sz": round(target_held_sz, 6),
            "order_sz": round(order_sz, 6),
            "order": order_result,
        }
        self.audit.log(audit_data)

        # 触发 Reef Observe 对齐
        try:
            from evolution.observer import get_observer
            get_observer().sync(client=trade_client, audit_log_path=str(self.audit.log_path))
        except Exception:
            pass

        return audit_data

    def start_auto_trade(
        self,
        strategy_path: str,
        inst_id: str,
        capital: float = 10000.0,
        leverage: int = 5,
        bar: str = "1H",
        max_position_pct: float = 0.30,
        interval_seconds: int = 3600,
        ladder_tp: bool = True,
    ) -> dict:
        """启动自动交易——按固定间隔循环执行信号。"""
        # 实盘安全门禁：检查策略分数
        if Config.is_live():
            try:
                st = load_strategy(strategy_path)
                score = st.get("best_score")
                if score is None or score <= 0:
                    return {
                        "ok": False,
                        "msg": f"拒绝启动实盘：策略评分过低 ({score} <= 0)，请先重新训练出正期望收益策略",
                    }
            except Exception as e:
                return {"ok": False, "msg": f"加载策略失败: {e}"}

        with self._lock:
            if self._auto_trade_state.get("running"):
                return {"ok": False, "msg": "自动交易已在运行中，请先停止"}
            self._auto_trade_state = {
                "running": True,
                "strategy_path": strategy_path,
                "inst_id": inst_id,
                "capital": capital,
                "leverage": leverage,
                "bar": bar,
                "max_position_pct": max_position_pct,
                "interval_seconds": interval_seconds,
                "ladder_tp": ladder_tp,
                "last_ladder_tp": None,
                "last_exit": None,
                "last_execute_time": None,
                "next_execute_time": time.time() + 2,  # 2 秒后首次执行
                "total_executions": 0,
                "total_orders": 0,
                "total_skips": 0,
                "signal_stats": {"long": 0, "short": 0, "flat": 0, "skip": 0, "error": 0},
                "signal_history": [],
                "last_result": None,
                "last_error": None,
                "started_at": time.time(),
            }

        # 启动后台线程
        t = threading.Thread(target=self._auto_trade_loop, daemon=True, name="auto-trade")
        t.start()
        self._auto_trade_thread = t

        self.audit.log({
            "event": "auto_trade_start",
            "inst_id": inst_id,
            "strategy_path": strategy_path,
            "bar": bar,
            "interval_seconds": interval_seconds,
            "capital": capital,
            "leverage": leverage,
        })

        return {"ok": True, "msg": f"自动交易已启动，间隔 {interval_seconds} 秒"}

    def stop_auto_trade(self) -> dict:
        """停止自动交易。"""
        with self._lock:
            if not self._auto_trade_state.get("running"):
                return {"ok": False, "msg": "自动交易未在运行"}
            self._auto_trade_state["running"] = False

        self.audit.log({"event": "auto_trade_stop"})

        return {"ok": True, "msg": "自动交易已停止"}

    def get_auto_trade_status(self) -> dict:
        """获取自动交易状态。"""
        state = self._auto_trade_state.copy()
        inst_id = state.get("inst_id")
        if inst_id:
            state["ladder_tp_ratchet"] = self._ladder_tp_ratchet.get(inst_id, 1.0)
            state["ladder_tp_base_sz"] = self._ladder_tp_base_sz.get(inst_id, 0.0)
            state["peak_price"] = self._peak_price.get(inst_id)
            state["breakeven_sl_active"] = (
                getattr(Config, "ENABLE_BREAKEVEN_SL", True)
                and self._ladder_tp_ratchet.get(inst_id, 1.0) <= Config.LADDER_TP_TIER1_CAP
            )
            state["trailing_sl_active"] = (
                getattr(Config, "ENABLE_TRAILING_SL", True)
                and self._ladder_tp_ratchet.get(inst_id, 1.0) <= Config.LADDER_TP_TIER2_CAP
            )
        if state.get("strategy_path"):
            state["strategy_name"] = pathlib.Path(state["strategy_path"]).name
        if state.get("next_execute_time"):
            state["next_execute_in"] = max(0, int(state["next_execute_time"] - time.time()))
        if state.get("started_at"):
            state["uptime_seconds"] = int(time.time() - state["started_at"])
        return state

    def _auto_trade_loop(self):
        """自动交易后台循环（在独立线程中运行）。"""
        while True:
            with self._lock:
                if not self._auto_trade_state.get("running"):
                    break
                next_time = self._auto_trade_state.get("next_execute_time", 0)
                interval = self._auto_trade_state.get("interval_seconds", 3600)

            # 等待到下一次执行时间
            now = time.time()
            if now < next_time:
                sleep_sec = min(next_time - now, 5)
                time.sleep(sleep_sec)
                # 盘中高频阶梯主动止盈监听（默认5秒一次，解耦长周期策略信号）
                with self._lock:
                    ladder_tp_enabled = self._auto_trade_state.get("ladder_tp", True)
                    curr_inst_id = self._auto_trade_state.get("inst_id")
                    is_running = self._auto_trade_state.get("running")
                if is_running and ladder_tp_enabled and curr_inst_id:
                    try:
                        self.check_realtime_ladder_tp(curr_inst_id)
                    except Exception:
                        pass
                continue

            # 执行信号
            with self._lock:
                if not self._auto_trade_state.get("running"):
                    break
                params = {
                    "strategy_path": self._auto_trade_state["strategy_path"],
                    "inst_id": self._auto_trade_state["inst_id"],
                    "capital": self._auto_trade_state["capital"],
                    "leverage": self._auto_trade_state["leverage"],
                    "bar": self._auto_trade_state["bar"],
                    "max_position_pct": self._auto_trade_state["max_position_pct"],
                    "ladder_tp": self._auto_trade_state.get("ladder_tp", True),
                }

            try:
                result = self.execute_signal(**params)
                with self._lock:
                    self._auto_trade_state["last_execute_time"] = time.time()
                    self._auto_trade_state["next_execute_time"] = time.time() + interval
                    self._auto_trade_state["total_executions"] += 1

                    # 提取信号信息
                    signal = result.get("signal", 0)
                    action = result.get("action", "空仓")
                    risk_passed = result.get("risk_passed", False)
                    order = result.get("order") or {}
                    ordered = risk_passed and bool(order.get("live") or order.get("simulated")) and not order.get("skipped")
                    skipped = not ordered

                    if ordered:
                        self._auto_trade_state["total_orders"] += 1
                    if skipped:
                        self._auto_trade_state["total_skips"] += 1

                    # 信号统计
                    stats = self._auto_trade_state["signal_stats"]
                    if skipped:
                        stats["skip"] += 1
                    elif signal > 0.05:
                        stats["long"] += 1
                    elif signal < -0.05:
                        stats["short"] += 1
                    else:
                        stats["flat"] += 1

                    # 信号历史（最多保留 30 条）
                    hist_entry = {
                        "time": time.time(),
                        "signal": round(signal, 4) if signal is not None else 0,
                        "action": action,
                        "price": result.get("last_price"),
                        "target_sz": result.get("target_held_sz", result.get("target_sz")),
                        "delta_sz": result.get("delta_sz", 0.0),
                        "ordered": ordered,
                        "skipped": skipped,
                        "risk_passed": risk_passed,
                    }
                    self._auto_trade_state["signal_history"].insert(0, hist_entry)
                    if len(self._auto_trade_state["signal_history"]) > 30:
                        self._auto_trade_state["signal_history"] = self._auto_trade_state["signal_history"][:30]

                    self._auto_trade_state["last_result"] = hist_entry
                    self._auto_trade_state["last_error"] = None
            except Exception as e:
                retry_wait = min(interval, 15)
                with self._lock:
                    self._auto_trade_state["last_execute_time"] = time.time()
                    self._auto_trade_state["next_execute_time"] = time.time() + retry_wait
                    self._auto_trade_state["total_executions"] += 1
                    self._auto_trade_state["signal_stats"]["error"] += 1
                    self._auto_trade_state["last_error"] = str(e)
                self.audit.log({
                    "event": "auto_trade_error",
                    "error": str(e),
                    "retry_in": retry_wait,
                })


# 全局单例
trading_service = TradingService()
