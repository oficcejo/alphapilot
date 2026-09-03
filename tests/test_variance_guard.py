"""
tests/test_variance_guard.py — 因子方差坍塌与退化公式防御测试
"""
import torch
import pytest
from model.vm import StackVM, validate_formula_structure
from model.engine import _is_degenerate_formula
from model.vocab import FORMULA_VOCAB
from api.services.trading_service import trading_service


def test_get_audit_log_exists():
    """验证 TradingService.get_audit_log 正常工作，不抛 AttributeError。"""
    logs = trading_service.get_audit_log(5)
    assert isinstance(logs, list)


def test_problematic_strategy_flagged_by_validator():
    """验证此前导致退化的线上 5m 公式被 validate_formula_structure 拦截。"""
    # AMIHUD_ILLIQ -> TS_SKEW_10 -> TS_QUANTILE_10 -> TS_ARG_MIN_5 -> TS_MAX_20 -> TS_RANK_10 -> JUMP -> ABS
    problem_formula = [41, 96, 95, 101, 98, 84, 73, 70]
    violations = validate_formula_structure(problem_formula, FORMULA_VOCAB.token_names)
    assert len(violations) >= 1
    # 必须检测到位置算子接极值算子或末尾恒正
    assert any("TS_ARG_MIN" in v or "ABS" in v or "恒正" in v for v in violations)


def test_problematic_strategy_flagged_by_engine():
    """验证 _is_degenerate_formula 能拦截问题公式。"""
    problem_formula = [41, 96, 95, 101, 98, 84, 73, 70]
    is_deg, reason = _is_degenerate_formula(problem_formula)
    assert is_deg is True
    assert "ABS" in reason or "恒正" in reason


def test_normalize_output_zeros_constant_windows():
    """验证 StackVM._normalize_output 对常数窗口输出严格 0.0，杜绝浮点微差伪信号。"""
    vm = StackVM()
    # 构造尾部 300 步完全恒定的张量
    x = torch.full((1, 800), 0.9236)
    norm = vm._normalize_output(x)
    # 全常数输入时输出应严格全为 0.0
    assert torch.all(norm == 0.0)
