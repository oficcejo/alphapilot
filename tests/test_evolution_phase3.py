"""
tests/test_evolution_phase3.py -- Reef Phase 3 (Grow, Shadow & Commit) 单元测试与端到端闭环
"""

import json
import shutil
import tempfile
import pathlib
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from config import Config
from model.vocab import FORMULA_VOCAB
from evolution.grow import GrowEngine
from evolution.shadow import ShadowEvaluator
from evolution.commit import CommitManager
from api.main import app


@pytest.fixture
def temp_phase3_dirs():
    temp_dir = tempfile.mkdtemp(prefix="test_phase3_")
    evo_dir = pathlib.Path(temp_dir) / "data" / "evolution"
    strat_dir = pathlib.Path(temp_dir) / "strategies"
    evo_dir.mkdir(parents=True, exist_ok=True)
    strat_dir.mkdir(parents=True, exist_ok=True)

    yield str(evo_dir), str(strat_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_grow_engine_mutation_and_feedback(temp_phase3_dirs):
    evo_dir, _ = temp_phase3_dirs
    engine = GrowEngine(data_dir=evo_dir)

    base_formula = [55, 120, 59, 124, 97, 104, 96, 97]
    mutated = engine.mutate_formula(base_formula)
    assert isinstance(mutated, list)
    assert len(mutated) > 0

    # 测试因果轨迹加权计算
    factors = torch.ones(1, 100) * -0.5  # 模拟在阴跌时看空/空仓
    timestamps = np.linspace(1788777297593, 1789196659360, 100).astype(np.int64)
    trajectories = [
        {
            "open_time_ms": 1788777297593,
            "close_time_ms": 1789044425941,
            "duration_hours": 74.2,
            "pnl_ratio": -0.157,
        }
    ]

    bonus = engine.compute_trajectory_bonus(factors, timestamps, trajectories)
    # 看空避开阴跌，应得到正向 bonus
    assert bonus > 0.0


def test_shadow_evaluator_5_gates(temp_phase3_dirs):
    evo_dir, _ = temp_phase3_dirs
    evaluator = ShadowEvaluator(data_dir=evo_dir)

    n_bars = 100
    n_feats = FORMULA_VOCAB.feature_count
    feat_tensor = torch.randn(1, n_feats, n_bars)
    target_ret = torch.randn(1, n_bars) * 0.01

    incumbent = {
        "formula": [0, 69],  # RET -> NEG
        "best_score": 5.0,
    }

    # 1. 测试退化常数因子 (std < 1e-4) -> Gate 2 拦截
    degenerate_cand = {
        "candidate_id": "cand_degen",
        "formula": [0],
        "score": 6.0,
    }
    with patch.object(evaluator.vm, "execute", return_value=torch.zeros(1, n_bars)):
        passed, report, comp = evaluator.evaluate_gates(degenerate_cand, incumbent, feat_tensor, target_ret)
        assert passed is False
        assert report["gate_non_degenerate"] is False

    # 2. 测试得分未超越基线 (score < 5.0 * 1.03) -> Gate 3 拦截
    low_score_cand = {
        "candidate_id": "cand_low_score",
        "formula": [0, 69],
        "score": 4.8,
    }
    with patch.object(evaluator.vm, "execute", return_value=torch.randn(1, n_bars)):
        passed, report, comp = evaluator.evaluate_gates(low_score_cand, incumbent, feat_tensor, target_ret)
        assert passed is False
        assert report["gate_score_improvement"] is False

    # 3. 测试通过全部门禁的优秀候选策略
    high_score_cand = {
        "candidate_id": "cand_winner",
        "formula": [55, 120, 59, 124],
        "score": 6.20,
    }
    with patch.object(evaluator.vm, "execute", return_value=torch.randn(1, n_bars)):
        passed, report, comp = evaluator.evaluate_gates(high_score_cand, incumbent, feat_tensor, target_ret)
        assert passed is True
        assert report["gate_passed"] is True
        assert report["gate_score_improvement"] is True
        assert report["gate_vm_compatibility"] is True


def test_commit_manager_atomic_swap(temp_phase3_dirs):
    evo_dir, strat_dir = temp_phase3_dirs
    commit_mgr = CommitManager(data_dir=evo_dir, strategies_dir=strat_dir)
    evaluator = ShadowEvaluator(data_dir=evo_dir)

    # 1. 初始化一个实盘基准策略文件
    target_strat_path = pathlib.Path(strat_dir) / "best_ETH-USDT-SWAP_1H.json"
    init_data = {
        "vocab_version": FORMULA_VOCAB.version,
        "symbol": "ETH-USDT-SWAP",
        "timeframe": "1H",
        "formula": [0, 69],
        "best_score": 5.819,
        "formula_decoded": "RET -> NEG",
    }
    target_strat_path.write_text(json.dumps(init_data), encoding="utf-8")

    # 2. 注册一个已通过门禁的候选策略进影子池
    cand = {
        "candidate_id": "cand_test_001",
        "symbol": "ETH-USDT-SWAP",
        "timeframe": "1H",
        "formula": [55, 120, 59, 124, 97],
        "formula_decoded": "SUPERTREND_DIR -> TS_ZSCORE_10 -> HURST_50 -> SIGMOID -> TS_MIN_20",
        "score": 6.18,
    }
    with patch("evolution.shadow.get_shadow_evaluator", return_value=evaluator):
        evaluator.register_candidate(cand, incumbent_strategy=init_data)
        # 强制标记为 PASSED
        evaluator.pool["cand_test_001"]["gate_report"]["gate_passed"] = True

        # 3. 执行交付
        res = commit_mgr.commit_candidate(
            candidate_id="cand_test_001",
            target_strategy_path=str(target_strat_path),
        )

        assert res["status"] == "success"
        assert res["new_score"] == 6.18
        assert res["previous_score"] == 5.819

        # 4. 验证旧文件已归档备份
        archive_files = list(commit_mgr.archive_dir.glob("*.json"))
        assert len(archive_files) == 1
        archived_content = json.loads(archive_files[0].read_text(encoding="utf-8"))
        assert archived_content["best_score"] == 5.819

        # 5. 验证目标实盘策略已被原子替换
        new_active = json.loads(target_strat_path.read_text(encoding="utf-8"))
        assert new_active["best_score"] == 6.18
        assert new_active["formula"] == [55, 120, 59, 124, 97]

        # 6. 验证提交审计日志 commits.jsonl
        assert commit_mgr.commits_file.exists()
        commits = commit_mgr.get_commit_history()
        assert len(commits) == 1
        assert commits[0]["candidate_id"] == "cand_test_001"
        assert commits[0]["new_score"] == 6.18


def test_api_phase3_endpoints():
    client = TestClient(app)

    # 1. GET /api/evolution/status 检查 Phase 3 全局状态
    res = client.get("/api/evolution/status")
    assert res.status_code == 200
    data = res.json()
    assert "components" in data
    assert "grow" in data["components"]
    assert "shadow" in data["components"]
    assert "commit" in data["components"]

    # 2. GET /api/evolution/shadow 检查影子池接口
    res_shadow = client.get("/api/evolution/shadow")
    assert res_shadow.status_code == 200
    assert "candidates" in res_shadow.json()

    # 3. GET /api/evolution/commits 检查交付审计历史接口
    res_commits = client.get("/api/evolution/commits")
    assert res_commits.status_code == 200
    assert "commits" in res_commits.json()
