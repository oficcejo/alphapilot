"""
evolution/grow.py -- Reef Strategy Grow & Evolution Engine (Phase 3: Strategy Evolution)

核心职责：
1. 实盘轨迹因果引导 (Trajectory-Guided Fitness):
   - 基于 Phase 1 沉淀的 trajectories.jsonl 赋予因果奖惩
   - 对 09-08~09-10 慢阴跌中空仓/做空的变异公式给予正向奖励
   - 对单边爆发中顺势跟随的变异公式给予顺势奖励
2. 在线公式变异与繁衍 (Formula Mutation & Breeding):
   - 继承基线优质公式 (如 best_ETH-USDT-SWAP_1H.json，评分 5.819)
   - 基于算子词表进行定向微调、算子替换与子树扩展
3. 异步后台进化调度 (Asynchronous Background Evolution):
   - 独立守护线程运行，零阻塞实盘交易
   - 产生高分候选策略后自动送入影子池 (Shadow Pool) 接受 5 重门禁评测
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import torch

from config import Config
from model.vocab import FORMULA_VOCAB
from model.vm import StackVM
from model.features import MT5FeatureEngineer
from data_pipeline.okx_client import get_public_client
from api.services.strategy_service import load_strategy, decode_formula

logger = logging.getLogger(__name__)


class GrowEngine:
    """Reef Grow 策略进化引擎。"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = pathlib.Path(data_dir or "data/evolution")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_file = self.data_dir / "trajectories.jsonl"

        self.vm = StackVM()
        self._lock = threading.Lock()
        self._is_running = False
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

        self._current_step = 0
        self._total_steps = 0
        self._best_score = -999.0
        self._best_candidate: Optional[dict] = None
        self._generated_candidates: List[dict] = []
        self._last_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running": self._is_running,
                "current_step": self._current_step,
                "total_steps": self._total_steps,
                "best_score": round(self._best_score, 4) if self._best_score > -900 else None,
                "best_candidate": self._best_candidate,
                "candidates_count": len(self._generated_candidates),
                "last_error": self._last_error,
            }

    # ── 1. 实盘因果轨迹奖惩权重提取 ────────────────────────────────────────

    def load_trajectories_feedback(self) -> List[dict]:
        """加载历史实盘因果反馈轨迹。"""
        if not self.trajectories_file.exists():
            return []
        items = []
        for line in self.trajectories_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
        return items

    def compute_trajectory_bonus(
        self,
        factors: torch.Tensor,
        candles_timestamps: np.ndarray,
        trajectories: List[dict],
    ) -> float:
        """根据历史实盘真实盈亏对候选因子的表现进行正负奖惩调整。

        例如：
        - 09-08~09-10 慢阴跌（duration > 36h 且亏损）：候选因子若在此区间保持空仓/看空，奖励 +0.5 ~ +1.0
        - 09-11~09-12 大涨（pnl > 10%）：候选因子若在此区间保持做多，奖励 +0.5 ~ +1.0
        """
        if not trajectories or candles_timestamps is None or len(candles_timestamps) == 0:
            return 0.0

        bonus = 0.0
        fac_1d = factors.squeeze().detach().cpu().numpy()
        n = min(len(fac_1d), len(candles_timestamps))
        if n < 10:
            return 0.0

        ts_series = candles_timestamps[:n]

        for tr in trajectories:
            open_ms = tr.get("open_time_ms", 0)
            close_ms = tr.get("close_time_ms", 0)
            pnl_ratio = tr.get("pnl_ratio", 0.0)
            duration_h = tr.get("duration_hours", 0.0)

            # 定位该交易在 K 线上的时间窗口
            mask = (ts_series >= open_ms) & (ts_series <= close_ms)
            if not np.any(mask):
                continue

            trade_factors = fac_1d[mask]
            avg_sig = float(np.tanh(trade_factors).mean())

            # 痛点 1：09-08~09-10 慢阴跌长时扛单亏损单
            if duration_h > 36.0 and pnl_ratio < -0.10:
                if avg_sig <= 0.15:  # 成功规避做多或提前出场
                    bonus += 0.8
                elif avg_sig > 0.40:  # 依然在此期间顽固做多
                    bonus -= 0.6

            # 痛点 2：09-11~09-12 大胜多单
            if pnl_ratio > 0.08:
                if avg_sig >= 0.30:  # 保持顺势做多
                    bonus += 0.6
                elif avg_sig < 0.0:  # 逆势做空
                    bonus -= 0.5

        return bonus

    # ── 2. 公式变异与生成 ──────────────────────────────────────────────────

    def mutate_formula(self, base_formula: List[int], mutation_rate: float = 0.25) -> List[int]:
        """对基础公式执行定向微调突变（算子替换、特征微调、局部插入）。"""
        formula = list(base_formula)
        f_count = FORMULA_VOCAB.feature_count
        o_count = len(FORMULA_VOCAB.operator_names)

        mutation_type = random.choice(["replace_op", "replace_feat", "tweak_tail", "substitute"])

        if mutation_type == "replace_op" and len(formula) > 1:
            # 替换其中的某个算子
            op_indices = [i for i, tok in enumerate(formula) if tok >= f_count]
            if op_indices:
                idx = random.choice(op_indices)
                new_op = random.randint(f_count, f_count + o_count - 1)
                formula[idx] = new_op

        elif mutation_type == "replace_feat" and len(formula) > 0:
            # 替换特征输入
            feat_indices = [i for i, tok in enumerate(formula) if tok < f_count]
            if feat_indices:
                idx = random.choice(feat_indices)
                new_feat = random.randint(0, f_count - 1)
                formula[idx] = new_feat

        elif mutation_type == "tweak_tail" and len(formula) > 2:
            # 变换尾部平滑/归一算子（如在 SIGMOID, TS_ZSCORE, TS_MIN, TS_MAX 间切换）
            last_op = random.randint(f_count, f_count + o_count - 1)
            formula[-1] = last_op

        else:
            # 交换相邻同类 token
            if len(formula) >= 3:
                i = random.randint(0, len(formula) - 2)
                formula[i], formula[i + 1] = formula[i + 1], formula[i]

        return formula

    # ── 3. 异步训练与进化主循环 ──────────────────────────────────────────

    def run_evolution_step(
        self,
        base_strategy: dict,
        feat_tensor: torch.Tensor,
        target_ret: torch.Tensor,
        candles_timestamps: np.ndarray,
        trajectories: List[dict],
    ) -> Optional[dict]:
        """单步变异与评估。"""
        base_formula = base_strategy.get("formula", [])
        if not base_formula:
            return None

        cand_formula = self.mutate_formula(base_formula)

        # 1. 语法与执行兼容性校验
        try:
            with torch.no_grad():
                factors = self.vm.execute(cand_formula, feat_tensor)
        except Exception:
            return None

        if factors is None or factors.numel() == 0:
            return None

        # 2. 方差与常数退化校验
        std = float(factors.std().item())
        if torch.isnan(factors).any() or torch.isinf(factors).any() or std < 1e-4:
            return None

        # 3. 计算基础风险调整收益 (Sortino / IC)
        pos = torch.tanh(factors)
        pnl = pos * target_ret - torch.abs(pos - torch.roll(pos, 1, dims=-1)) * 0.0008
        downside = pnl[pnl < 0]
        downside_std = float(downside.std().item()) if downside.numel() > 0 else float(pnl.std().item())
        if downside_std < 1e-6:
            downside_std = 1e-6

        # 年化 Sortino
        sortino = float(pnl.mean().item() / downside_std * (6240 ** 0.5))
        sortino = max(-5.0, min(15.0, sortino))

        # 4. 加入 Phase 1 实盘历史因果加权
        bonus = self.compute_trajectory_bonus(factors, candles_timestamps, trajectories)
        final_score = round(sortino * 0.7 + bonus, 4)

        candidate = {
            "candidate_id": f"cand_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}",
            "symbol": base_strategy.get("symbol", "ETH-USDT-SWAP"),
            "timeframe": base_strategy.get("timeframe", "1H"),
            "formula": cand_formula,
            "formula_decoded": decode_formula(cand_formula),
            "score": final_score,
            "raw_sortino": round(sortino, 4),
            "feedback_bonus": round(bonus, 4),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_strategy": base_strategy.get("formula_decoded", "base"),
        }

        return candidate

    def _evolution_worker(
        self,
        strategy_path: str,
        inst_id: str,
        bar: str,
        steps: int,
    ) -> None:
        """后台进化工作线程。"""
        try:
            with self._lock:
                self._is_running = True
                self._stop_flag = False
                self._current_step = 0
                self._total_steps = steps
                self._last_error = None

            # 1. 加载基准策略
            base_strategy = load_strategy(strategy_path)
            base_score = float(base_strategy.get("best_score", 0.0))
            self._best_score = base_score

            # 2. 加载近期 K 线与因果特征
            min_bars = getattr(Config, "REALTIME_MIN_BARS", 800)
            client = get_public_client()
            candles = client.get_recent_candles(inst_id, bar=bar, total=min_bars, only_confirmed=True)
            if not candles or len(candles) < 200:
                raise RuntimeError(f"获取 {inst_id} {bar} 行情不足，无法进行策略进化")

            close_arr = np.array([float(c[4]) for c in candles], dtype=np.float64)
            open_arr = np.array([float(c[1]) for c in candles], dtype=np.float64)
            high_arr = np.array([float(c[2]) for c in candles], dtype=np.float64)
            low_arr = np.array([float(c[3]) for c in candles], dtype=np.float64)
            vol_arr = np.array([float(c[5]) for c in candles], dtype=np.float64)
            time_arr = np.array([int(c[0]) for c in candles], dtype=np.int64)

            raw_dict = {
                "close": torch.from_numpy(close_arr).unsqueeze(0).float(),
                "open": torch.from_numpy(open_arr).unsqueeze(0).float(),
                "high": torch.from_numpy(high_arr).unsqueeze(0).float(),
                "low": torch.from_numpy(low_arr).unsqueeze(0).float(),
                "volume": torch.from_numpy(vol_arr).unsqueeze(0).float(),
                "time": time_arr,
            }
            feat_tensor = MT5FeatureEngineer.compute_features(raw_dict)

            # 严格因果 target_ret: log(open[t+2] / open[t+1])
            open_t = raw_dict["open"]
            target_ret = torch.zeros_like(open_t)
            if target_ret.shape[1] >= 3:
                num = open_t[:, 2:]
                den = open_t[:, 1:-1].clone()
                den[den == 0] = 1.0
                target_ret[:, :-2] = torch.log(num / den)

            trajectories = self.load_trajectories_feedback()

            # 3. 进化主循环
            for step in range(steps):
                if self._stop_flag:
                    break

                cand = self.run_evolution_step(
                    base_strategy=base_strategy,
                    feat_tensor=feat_tensor,
                    target_ret=target_ret,
                    candles_timestamps=time_arr,
                    trajectories=trajectories,
                )

                with self._lock:
                    self._current_step = step + 1
                    if cand is not None:
                        if cand["score"] > self._best_score:
                            self._best_score = cand["score"]
                            self._best_candidate = cand
                            self._generated_candidates.append(cand)
                            # 自动注册进影子评估池
                            try:
                                from evolution.shadow import get_shadow_evaluator
                                get_shadow_evaluator().register_candidate(cand, incumbent_strategy=base_strategy)
                            except Exception as ex:
                                logger.warning(f"送入影子池失败: {ex}")

                # 微睡让出 CPU，避免打满资源
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"GrowEngine 进化异常: {e}", exc_info=True)
            with self._lock:
                self._last_error = str(e)
        finally:
            with self._lock:
                self._is_running = False

    def start_grow(
        self,
        strategy_path: str = "strategies/best_ETH-USDT-SWAP_1H.json",
        inst_id: str = "ETH-USDT-SWAP",
        bar: str = "1H",
        steps: int = 150,
    ) -> dict:
        """启动后台策略进化任务。"""
        with self._lock:
            if self._is_running:
                return {"status": "error", "message": "已有进化任务在运行中"}

        self._thread = threading.Thread(
            target=self._evolution_worker,
            args=(strategy_path, inst_id, bar, steps),
            daemon=True,
            name="ReefGrowWorker",
        )
        self._thread.start()
        return {
            "status": "started",
            "strategy_path": strategy_path,
            "inst_id": inst_id,
            "bar": bar,
            "steps": steps,
        }

    def stop_grow(self) -> dict:
        """停止后台进化任务。"""
        with self._lock:
            if not self._is_running:
                return {"status": "ok", "message": "当前无运行中的进化任务"}
            self._stop_flag = True
        return {"status": "stopping", "message": "已发送终止信号"}


# 单例管理
_global_grow_engine: Optional[GrowEngine] = None
_global_grow_lock = threading.Lock()


def get_grow_engine() -> GrowEngine:
    global _global_grow_engine
    if _global_grow_engine is None:
        with _global_grow_lock:
            if _global_grow_engine is None:
                _global_grow_engine = GrowEngine()
    return _global_grow_engine
