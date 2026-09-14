"""
evolution/commit.py -- Reef Atomic Commit & Strategy Hot-Swap Manager (Phase 3: Safe Delivery)

核心职责：
1. 策略归档备份 (Safety Archive):
   - 交付前自动将当前实盘旧策略完整备份至 strategies/archive/
   - 附加时间戳与历史评分，保证任何时刻均可 100% 回滚
2. 原子热替换 (Atomic Hot-Swap):
   - 原子化更新目标策略文件 (如 best_ETH-USDT-SWAP_1H.json)
   - 适配 TradingService 的即时文件加载机制，无需重启服务即可下一 Tick 自动生效
3. 交付审计日志沉淀 (Commit Audit Store):
   - 记录每次策略迭代的因果来源、新旧得分、公式差异至 commits.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List

from config import Config
from model.vocab import FORMULA_VOCAB
from api.services.strategy_service import load_strategy

logger = logging.getLogger(__name__)


class CommitManager:
    """Reef 原子交付与策略热切换管理器。"""

    def __init__(self, data_dir: Optional[str] = None, strategies_dir: Optional[str] = None):
        self.data_dir = pathlib.Path(data_dir or "data/evolution")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.commits_file = self.data_dir / "commits.jsonl"

        self.strategies_dir = pathlib.Path(strategies_dir or "strategies")
        self.archive_dir = self.strategies_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

    def get_commit_history(self, limit: int = 50) -> List[dict]:
        """获取历史策略交付记录。"""
        if not self.commits_file.exists():
            return []
        lines = [l.strip() for l in self.commits_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        commits = []
        for line in reversed(lines):
            try:
                commits.append(json.loads(line))
                if len(commits) >= limit:
                    break
            except Exception:
                pass
        return commits

    def commit_candidate(
        self,
        candidate_id: str,
        target_strategy_path: Optional[str] = None,
        force: bool = False,
    ) -> dict:
        """执行候选策略的原子交付与上线。"""
        from evolution.shadow import get_shadow_evaluator
        evaluator = get_shadow_evaluator()
        candidate = evaluator.get_candidate(candidate_id)

        if not candidate:
            raise ValueError(f"候选策略 {candidate_id} 在影子池中不存在")

        gate_report = candidate.get("gate_report", {})
        gate_passed = gate_report.get("gate_passed", False)

        if not gate_passed and not force:
            raise ValueError(
                f"候选策略 {candidate_id} 未通过 5 重门禁核验，禁止上线交付: {gate_report.get('reasons')}"
            )

        symbol = candidate.get("symbol", "ETH-USDT-SWAP")
        timeframe = candidate.get("timeframe", "1H")

        if target_strategy_path:
            target_path = pathlib.Path(target_strategy_path)
        else:
            target_path = self.strategies_dir / f"best_{symbol}_{timeframe}.json"

        with self._lock:
            archived_path = None
            prev_score = None
            prev_formula_decoded = None

            # 1. 归档当前旧策略
            if target_path.exists():
                try:
                    prev_strat = load_strategy(str(target_path))
                    prev_score = prev_strat.get("best_score")
                    prev_formula_decoded = prev_strat.get("formula_decoded")
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_name = f"{target_path.stem}_{ts_str}_score{prev_score or 0:.2f}.json"
                    archived_path = self.archive_dir / archive_name
                    shutil.copy2(target_path, archived_path)
                except Exception as ex:
                    logger.warning(f"策略归档警告: {ex}")

            # 2. 组装新策略数据
            commit_id = f"cmt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
            new_strategy_data = {
                "vocab_version": FORMULA_VOCAB.version,
                "symbol": symbol,
                "timeframe": timeframe,
                "formula": candidate.get("formula"),
                "best_score": candidate.get("score"),
                "formula_decoded": candidate.get("formula_decoded"),
                "committed_at": datetime.now(timezone.utc).isoformat(),
                "commit_id": commit_id,
                "candidate_id": candidate_id,
            }

            # 3. 原子写入目标策略文件
            tmp_path = target_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(new_strategy_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            shutil.move(str(tmp_path), str(target_path))

            # 4. 沉淀提交审计日志
            commit_event = {
                "commit_id": commit_id,
                "candidate_id": candidate_id,
                "target_file": str(target_path),
                "archived_file": str(archived_path) if archived_path else None,
                "previous_score": prev_score,
                "previous_formula": prev_formula_decoded,
                "new_score": candidate.get("score"),
                "new_formula": candidate.get("formula_decoded"),
                "gate_report": gate_report,
                "committed_at": datetime.now(timezone.utc).isoformat(),
                "auto_committed": getattr(Config, "AUTO_COMMIT_STRATEGY", False),
            }

            with open(self.commits_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(commit_event, ensure_ascii=False) + "\n")

            # 5. 更新影子池状态
            evaluator.update_candidate_status(candidate_id, "COMMITTED")

            logger.info(f"Reef Commit: 策略成功交付上线 -> {target_path} (Score: {prev_score} -> {candidate.get('score')})")

            return {
                "status": "success",
                "commit_id": commit_id,
                "target_strategy": str(target_path),
                "archived_backup": str(archived_path) if archived_path else None,
                "previous_score": prev_score,
                "new_score": candidate.get("score"),
                "new_formula": candidate.get("formula_decoded"),
            }


# 单例管理
_global_commit_manager: Optional[CommitManager] = None
_global_commit_lock = threading.Lock()


def get_commit_manager() -> CommitManager:
    global _global_commit_manager
    if _global_commit_manager is None:
        with _global_commit_lock:
            if _global_commit_manager is None:
                _global_commit_manager = CommitManager()
    return _global_commit_manager
