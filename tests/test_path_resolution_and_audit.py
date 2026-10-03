"""
tests/test_path_resolution_and_audit.py -- 路径解析、策略加载与审计日志目录兼容性测试
"""

import os
import json
import shutil
import tempfile
import pathlib
import pytest

from config import Config, resolve_path
from api.services.strategy_service import load_strategy, delete_strategy, strategies_dir
from api.services.trading_service import AuditLog
from evolution.shadow import ShadowEvaluator


def test_resolve_path():
    """测试相对路径统一解析为基于 BASE_DIR 的绝对路径。"""
    base = pathlib.Path(Config.BASE_DIR)
    
    # 相对路径解析
    rel = "strategies/test_strategy.json"
    resolved = resolve_path(rel)
    assert resolved == base / rel
    assert resolved.is_absolute()
    
    # 绝对路径保持不变
    abs_path = (base / "strategies" / "test_strategy.json").resolve()
    assert resolve_path(abs_path) == abs_path
    
    # Config.resolve_path 静态方法一致
    assert Config.resolve_path(rel) == resolved


def test_load_strategy_relative_path_and_diagnostics():
    """测试在不同工作目录下加载策略，以及缺失策略时的诊断信息。"""
    # 1. 验证真实存在的策略能通过相对路径加载
    strat = load_strategy("strategies/best_ETH-USDT-SWAP_1H.json")
    assert strat["symbol"] == "ETH-USDT-SWAP"
    assert "formula" in strat
    assert len(strat["formula"]) == 8

    # 2. 验证仅传入文件名也能加载
    strat_by_name = load_strategy("best_ETH-USDT-SWAP_1H.json")
    assert strat_by_name["symbol"] == "ETH-USDT-SWAP"

    # 3. 验证在切换工作目录后依然能正确加载
    old_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix="test_cwd_")
    try:
        os.chdir(tmp_dir)
        # 在非项目根目录下加载相对路径
        strat_non_cwd = load_strategy("strategies/best_ETH-USDT-SWAP_1H.json")
        assert strat_non_cwd["symbol"] == "ETH-USDT-SWAP"

        # 4. 测试文件不存在时的诊断信息包含 BASE_DIR 与 CWD
        with pytest.raises(FileNotFoundError) as exc_info:
            load_strategy("non_existent_strategy_123.json")
        err_msg = str(exc_info.value)
        assert "non_existent_strategy_123.json" in err_msg
        assert "BASE_DIR=" in err_msg
        assert "CWD=" in err_msg
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_audit_log_directory_fallback():
    """测试宿主机或 Docker 误将审计日志挂载为目录时的防崩兜底逻辑。"""
    tmp_dir = tempfile.mkdtemp(prefix="test_audit_")
    try:
        # 模拟 Docker 把 trading_audit.jsonl 建成了一个目录
        fake_mount_dir = pathlib.Path(tmp_dir) / "trading_audit.jsonl"
        fake_mount_dir.mkdir(parents=True, exist_ok=True)

        audit = AuditLog(log_path=str(fake_mount_dir))
        
        # 记录审计事件，不应抛出 IsADirectoryError
        audit.log({"action": "TEST_OPEN", "price": 100.0})
        
        # 读取最近记录
        records = audit.get_recent(10)
        assert len(records) == 1
        assert records[0]["action"] == "TEST_OPEN"
        assert records[0]["price"] == 100.0

        # 验证文件被安全保存在子目录中
        inner_file = fake_mount_dir / "trading_audit.jsonl"
        assert inner_file.exists()
        assert inner_file.is_file()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_shadow_evaluator_missing_baseline_rejection():
    """测试基准策略缺失时影子评测应明确拒绝，不得使用虚假公式。"""
    tmp_dir = tempfile.mkdtemp(prefix="test_shadow_")
    try:
        evaluator = ShadowEvaluator(data_dir=tmp_dir)
        
        candidate = {
            "candidate_id": "cand_999",
            "formula": [1, 2, 3],
            "score": 6.0,
        }
        
        # 传入一个完全不存在的基准策略测试逻辑，或验证 register_candidate 不使用虚假 4-token 公式
        # 当找不到有效基准策略时（通过 mock 或指定不存在的 STRATEGY_FILE）
        from unittest.mock import patch
        with patch("evolution.shadow.load_strategy", side_effect=FileNotFoundError("Mock Not Found")):
            result = evaluator.register_candidate(candidate)
            assert result["status"] == "REJECTED"
            assert not result["gate_report"]["gate_passed"]
            assert any("未找到有效基准策略" in r for r in result["gate_report"]["reasons"])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
