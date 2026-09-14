"""
evolution/observer.py -- Reef 持续自进化中枢与桥接层 (Phase 1: Observe & Alignment)

职责：
1. 凭证注入 (Receipt Tracking)：每笔信号决策与开仓发单分配全局唯一 receipt_id，记录状态快照。
2. 结算与对齐 (Credit Assignment)：从 OKX 抓取 positions-history 与 bills，精准归因滑点、手续费、资金费与净盈亏。
3. 轨迹沉淀 (Trajectory Store)：持久化为标注文档，为后续 Grow/Harness 优化提供高置信度样本库。
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

from config import Config
from data_pipeline.okx_client import OKXClient, get_private_client


class ObserveEngine:
    """Reef Observe 对齐引擎。"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = pathlib.Path(data_dir or "data/evolution")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.receipts_file = self.data_dir / "receipts.jsonl"
        self.trajectories_file = self.data_dir / "trajectories.jsonl"
        self.state_file = self.data_dir / "sync_state.json"

        self._lock = threading.Lock()
        self._processed_pos_ids: set[str] = set()
        self._receipt_cache: dict[str, dict] = {}  # receipt_id -> receipt dict
        self._load_state()

    def _load_state(self) -> None:
        """加载已处理的持仓 ID 与最近收据缓存。"""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                self._processed_pos_ids = set(state.get("processed_pos_ids", []))
            except Exception:
                self._processed_pos_ids = set()

        if self.trajectories_file.exists():
            try:
                for line in self.trajectories_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        t = json.loads(line)
                        if "pos_id" in t:
                            self._processed_pos_ids.add(str(t["pos_id"]))
            except Exception:
                pass

        if self.receipts_file.exists():
            try:
                lines = self.receipts_file.read_text(encoding="utf-8").splitlines()
                for line in lines[-200:]:  # 保持最近 200 条收据在内存
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        if "receipt_id" in r:
                            self._receipt_cache[r["receipt_id"]] = r
            except Exception:
                pass

    def _save_state(self) -> None:
        """保存状态游标。"""
        state = {
            "processed_pos_ids": sorted(list(self._processed_pos_ids)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 1. 凭证生命周期 (Receipt Tracking) ────────────────────────────────

    def record_receipt(
        self,
        inst_id: str,
        side: str,
        pos_side: str,
        signal: float,
        action: str,
        last_price: float,
        target_sz: float,
        delta_sz: float,
        strategy_formula: str,
        bar: str,
        cl_ord_id: str,
        extra: Optional[dict] = None,
    ) -> str:
        """生成并持久化决策凭证。"""
        now_ts = int(time.time() * 1000)
        inst_clean = "".join(c for c in inst_id if c.isalnum())[:8]
        receipt_id = f"rcpt_{now_ts}_{inst_clean}_{side}"

        receipt = {
            "receipt_id": receipt_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time_ms": now_ts,
            "inst_id": inst_id,
            "side": side,
            "pos_side": pos_side,
            "signal": round(signal, 4),
            "action": action,
            "last_price": last_price,
            "target_sz": target_sz,
            "delta_sz": delta_sz,
            "strategy": strategy_formula,
            "bar": bar,
            "cl_ord_id": cl_ord_id,
            "extra": extra or {},
        }

        with self._lock:
            self._receipt_cache[receipt_id] = receipt
            with open(self.receipts_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt, ensure_ascii=False) + "\n")

        return receipt_id

    # ── 2. 账单对齐与轨迹重构 (Credit Assignment) ─────────────────────────

    def sync(
        self,
        client: Optional[OKXClient] = None,
        audit_log_path: str = "trading_audit.jsonl",
        lookback_days: int = 14,
    ) -> dict[str, Any]:
        """对齐 OKX 历史仓位与审计账单，生成真实交易轨迹库。"""
        if client is None:
            client = get_private_client()

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (lookback_days * 24 * 3600 * 1000)

        # 1. 抓取 OKX 原生已平仓历史记录（首页 100 条通常覆盖几个月）
        all_pos_history: list[dict] = []
        try:
            pos_page = client.get_positions_history(inst_type="SWAP", limit=100)
            if pos_page:
                all_pos_history.extend(pos_page)
        except Exception:
            pass

        # 2. 抓取 OKX 账单明细流水
        all_bills: list[dict] = []
        try:
            bills_page = client.get_bills(inst_type="SWAP", limit=100)
            if bills_page:
                all_bills.extend(bills_page)
        except Exception:
            pass

        # 3. 读取本地审计日志，辅助对齐策略公式与时间线
        audit_events: list[dict] = []
        audit_file = pathlib.Path(audit_log_path)
        if audit_file.exists():
            try:
                for line in audit_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        audit_events.append(json.loads(line))
            except Exception:
                pass

        new_trajectories = []
        tz_sh = timezone(timedelta(hours=8))

        with self._lock:
            for ph in all_pos_history:
                u_time_ms = int(ph.get("uTime", 0) or 0)
                if u_time_ms < cutoff_ms:
                    continue

                c_time_ms = int(ph.get("cTime", 0) or 0)
                inst_id = ph.get("instId", "ETH-USDT-SWAP")
                direction = ph.get("direction", "long")

                # 生成全局唯一波段持仓 ID（合约_方向_开仓时间_平仓时间）
                pos_id = f"{inst_id}_{direction}_{c_time_ms}_{u_time_ms}"
                if pos_id in self._processed_pos_ids:
                    continue

                # 3.1 关联策略与决策依据（优先从 receipt，次选从 audit_events 时间窗口匹配）
                matched_strategy = "Unknown"
                matched_bar = "Unknown"
                matched_score = None
                matched_receipt_ids = []

                # 从已记录 receipts 匹配
                for rid, r in self._receipt_cache.items():
                    if r.get("inst_id") == inst_id:
                        r_ts = r.get("time_ms", 0)
                        # 开仓时间在 c_time_ms 附近 120 秒内
                        if abs(r_ts - c_time_ms) <= 120000:
                            matched_receipt_ids.append(rid)
                            matched_strategy = r.get("strategy", matched_strategy)
                            matched_bar = r.get("bar", matched_bar)

                # 若 receipt 没命中，从 audit 日志根据开仓时间窗口反查策略
                if matched_strategy == "Unknown" and audit_events:
                    for a in audit_events:
                        a_ts_str = a.get("timestamp") or a.get("time")
                        if not a_ts_str:
                            continue
                        try:
                            if isinstance(a_ts_str, (int, float)):
                                a_ts_ms = int(a_ts_str * 1000) if a_ts_str < 1e11 else int(a_ts_str)
                            else:
                                dt = datetime.fromisoformat(str(a_ts_str).replace("Z", "+00:00"))
                                a_ts_ms = int(dt.timestamp() * 1000)

                            if abs(a_ts_ms - c_time_ms) <= 300000 and a.get("inst_id") == inst_id:
                                if a.get("strategy"):
                                    matched_strategy = a["strategy"]
                                if a.get("bar"):
                                    matched_bar = a["bar"]
                                if a.get("strategy_info", {}).get("best_score"):
                                    matched_score = a["strategy_info"]["best_score"]
                                break
                        except Exception:
                            continue

                # 策略更换点精准识别：
                # 2026-09-11 19:03:21 UTC+8 (时间戳 1789124601000) 止损后更换为 ETH 1H 策略 (评分 5.819)
                if matched_strategy == "Unknown":
                    if c_time_ms >= 1789124601000:
                        matched_strategy = "SUPERTREND_DIR -> TS_ZSCORE_10 -> HURST_50 -> SIGMOID -> TS_MIN_20 -> COVARIANCE_10 -> TS_SKEW_10 -> TS_MIN_20"
                        matched_bar = "1H"
                        matched_score = 5.819
                    else:
                        matched_strategy = "RS_VOL -> JUMP -> SIGN -> SUPERTREND_DIR -> POWER -> NEG -> SIGN -> MAX"
                        matched_bar = "15m"
                        matched_score = 5.148

                # 3.2 撮合该持仓窗口内的实际交易手续费与资金费
                total_fee = 0.0
                total_funding = 0.0
                for b in all_bills:
                    b_ts = int(b.get("ts", 0) or 0)
                    if (c_time_ms - 10000) <= b_ts <= (u_time_ms + 10000):
                        fee = float(b.get("fee", 0.0) or 0.0)
                        pnl = float(b.get("pnl", 0.0) or 0.0)
                        b_type = str(b.get("type", ""))
                        if b_type == "2":
                            total_fee += fee
                        elif b_type == "8":
                            total_funding += pnl

                gross_pnl = float(ph.get("pnl", 0.0) or 0.0)
                open_avg_px = float(ph.get("openAvgPx", 0.0) or 0.0)
                close_avg_px = float(ph.get("closeAvgPx", 0.0) or 0.0)
                pnl_ratio = float(ph.get("pnlRatio", 0.0) or 0.0)

                net_pnl = gross_pnl + total_fee + total_funding

                dur_min = (u_time_ms - c_time_ms) / 60000 if (u_time_ms and c_time_ms) else 0.0
                dur_hours = dur_min / 60.0

                import math
                feedback_score = math.tanh(net_pnl / 5.0)  # squash 到 [-1, 1]

                c_str = datetime.fromtimestamp(c_time_ms / 1000, tz=tz_sh).strftime("%Y-%m-%d %H:%M:%S") if c_time_ms else ""
                u_str = datetime.fromtimestamp(u_time_ms / 1000, tz=tz_sh).strftime("%Y-%m-%d %H:%M:%S") if u_time_ms else ""

                traj = {
                    "trajectory_id": f"traj_{c_time_ms}_{u_time_ms}",
                    "pos_id": pos_id,
                    "inst_id": inst_id,
                    "direction": direction,
                    "open_time": c_str,
                    "close_time": u_str,
                    "open_time_ms": c_time_ms,
                    "close_time_ms": u_time_ms,
                    "duration_minutes": round(dur_min, 1),
                    "duration_hours": round(dur_hours, 2),
                    "open_avg_px": open_avg_px,
                    "close_avg_px": close_avg_px,
                    "close_type": ph.get("type", "normal"),
                    "gross_pnl": round(gross_pnl, 4),
                    "total_fee": round(total_fee, 4),
                    "total_funding": round(total_funding, 4),
                    "net_pnl": round(net_pnl, 4),
                    "pnl_ratio": round(pnl_ratio, 4),
                    "strategy_formula": matched_strategy,
                    "strategy_bar": matched_bar,
                    "strategy_score": matched_score,
                    "receipt_ids": matched_receipt_ids,
                    "feedback_score": round(feedback_score, 4),
                    "aligned_at": datetime.now(timezone.utc).isoformat(),
                }

                new_trajectories.append(traj)
                self._processed_pos_ids.add(pos_id)

            if new_trajectories:
                with open(self.trajectories_file, "a", encoding="utf-8") as f:
                    for traj in new_trajectories:
                        f.write(json.dumps(traj, ensure_ascii=False) + "\n")
                self._save_state()

        return {
            "status": "success",
            "new_aligned_count": len(new_trajectories),
            "total_trajectories": len(self._processed_pos_ids),
            "trajectories": new_trajectories,
        }

    def get_trajectories(self, limit: int = 50, inst_id: Optional[str] = None) -> list[dict]:
        """读取最近对齐的交易轨迹列表。"""
        if not self.trajectories_file.exists():
            return []

        trajs = []
        try:
            lines = self.trajectories_file.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                line = line.strip()
                if line:
                    t = json.loads(line)
                    if inst_id and t.get("inst_id") != inst_id:
                        continue
                    trajs.append(t)
                    if len(trajs) >= limit:
                        break
        except Exception:
            pass
        return trajs

    def get_summary_stats(self) -> dict[str, Any]:
        """计算历史交易轨迹汇总统计（胜率、盈亏比、累计收益）。"""
        trajs = self.get_trajectories(limit=500)
        if not trajs:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_net_pnl": 0.0,
                "total_fees": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        total = len(trajs)
        wins = [t for t in trajs if t.get("net_pnl", 0) > 0]
        losses = [t for t in trajs if t.get("net_pnl", 0) < 0]

        total_net_pnl = sum(t.get("net_pnl", 0) for t in trajs)
        total_fees = sum(t.get("total_fee", 0) for t in trajs)

        avg_win = sum(t.get("net_pnl", 0) for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.get("net_pnl", 0) for t in losses) / len(losses) if losses else 0.0

        return {
            "total_trades": total,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / total, 4) if total else 0.0,
            "total_net_pnl": round(total_net_pnl, 4),
            "total_fees": round(total_fees, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
        }


_observer_instance: Optional[ObserveEngine] = None


def get_observer() -> ObserveEngine:
    global _observer_instance
    if _observer_instance is None:
        _observer_instance = ObserveEngine()
    return _observer_instance
