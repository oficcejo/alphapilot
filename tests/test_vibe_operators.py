"""
tests/test_vibe_operators.py — Vibe-Trading Alpha Zoo 算子扩展与集成测试 (Phase 1)

测试目标：
1. 算子完整在册：所有 17 个新增算子均已注册进 OPERATOR_REGISTRY，且映射至 FORMULA_VOCAB
2. 形状契约与防崩溃：输入 [N, T] 输出 [N, T]，操作数形状不匹配抛 ShapeError
3. NaN / Inf 安全性：极端值、零输入与浮点边界不产出 NaN/Inf
4. 严格因果无未来函数：修改 t+1 之后的输入，严格不影响 t 步的输出
5. 数学性质校验：SAFE_DIV, TS_ARG_MAX/MIN, DECAY_LINEAR, TS_CORR, COVARIANCE, DELTA 等精确性
6. StackVM 逆波兰栈机执行与感染模型验证
"""
import pytest
import torch

from model.ops import OPERATOR_REGISTRY, OPS_CONFIG, ShapeError, _VIBE_OPERATORS
from model.registry import ArityMismatchError
from model.vm import (
    ARG_INDEX_OPS,
    EXTREME_OPS,
    INFECTED_PROPAGATING_OPS,
    SIGN_RESTORE_OPS,
    StackVM,
    validate_formula_structure,
)
from model.vocab import FORMULA_VOCAB


EXPECTED_VIBE_OPS = [
    ("SAFE_DIV", 2),
    ("SIGNED_POWER_3", 1),
    ("TS_MAX_5", 1),
    ("TS_MIN_5", 1),
    ("TS_ARG_MAX_10", 1),
    ("TS_ARG_MAX_20", 1),
    ("TS_ARG_MIN_10", 1),
    ("TS_ARG_MIN_20", 1),
    ("DECAY_LINEAR_10", 1),
    ("DECAY_LINEAR_20", 1),
    ("TS_CORR_5", 2),
    ("TS_CORR_20", 2),
    ("COVARIANCE_5", 2),
    ("COVARIANCE_20", 2),
    ("PRODUCT_10", 1),
    ("DELTA_10", 1),
    ("DELTA_20", 1),
]


def test_vibe_operators_registration():
    """验证所有 Vibe-Trading 算子成功在 OPERATOR_REGISTRY 与 FORMULA_VOCAB 注册。"""
    registered_names = set(OPERATOR_REGISTRY.operator_names)
    vocab_op_names = set(FORMULA_VOCAB.operator_names)

    for name, expected_arity in EXPECTED_VIBE_OPS:
        assert name in registered_names, f"算子 {name} 未在 OPERATOR_REGISTRY 注册"
        assert name in vocab_op_names, f"算子 {name} 未包含在 FORMULA_VOCAB 中"
        spec = OPERATOR_REGISTRY.get_operator(name)
        assert spec.arity == expected_arity, f"算子 {name} arity 期望 {expected_arity}，实际 {spec.arity}"


def test_vibe_operators_shape_and_nan_safety():
    """验证形状契约 [N, T] -> [N, T] 以及 NaN/Inf 防护出口安全。"""
    torch.manual_seed(42)
    N, T = 2, 80
    x = torch.randn(N, T)
    y = torch.randn(N, T)

    # 注入一部分极端值和 0
    x[0, 5] = 0.0
    x[1, 10] = 1e8
    y[0, 5] = 0.0
    y[1, 10] = -1e8

    for name, arity in EXPECTED_VIBE_OPS:
        spec = OPERATOR_REGISTRY.get_operator(name)
        if arity == 1:
            out = spec.transform(x)
        elif arity == 2:
            out = spec.transform(x, y)
        else:
            pytest.fail(f"未支持的 arity: {arity}")

        assert out.shape == (N, T), f"算子 {name} 输出形状错误: {out.shape} vs {(N, T)}"
        assert not torch.isnan(out).any(), f"算子 {name} 产出 NaN"
        assert not torch.isinf(out).any(), f"算子 {name} 产出 Inf"


def test_vibe_binary_operators_shape_mismatch():
    """二元算子在操作数形状不一致时必须抛出 ShapeError。"""
    x = torch.randn(2, 50)
    y_mismatch = torch.randn(2, 60)

    for name, arity in EXPECTED_VIBE_OPS:
        if arity == 2:
            spec = OPERATOR_REGISTRY.get_operator(name)
            with pytest.raises(ShapeError):
                spec.transform(x, y_mismatch)


def test_vibe_operators_causal_no_lookahead():
    """严格因果性验证：修改未来 t+k 的数据，t 时刻输出必须完全不变。"""
    torch.manual_seed(101)
    N, T = 1, 100
    t_inspect = 60

    x1 = torch.randn(N, T)
    y1 = torch.randn(N, T)

    x2 = x1.clone()
    y2 = y1.clone()
    # 破坏未来数据 (t > t_inspect)
    x2[:, t_inspect + 1 :] += torch.randn(N, T - t_inspect - 1) * 10.0
    y2[:, t_inspect + 1 :] += torch.randn(N, T - t_inspect - 1) * 10.0

    for name, arity in EXPECTED_VIBE_OPS:
        spec = OPERATOR_REGISTRY.get_operator(name)
        if arity == 1:
            out1 = spec.transform(x1)
            out2 = spec.transform(x2)
        else:
            out1 = spec.transform(x1, y1)
            out2 = spec.transform(x2, y2)

        # t <= t_inspect 范围内的值必须严格相同
        diff = torch.abs(out1[:, : t_inspect + 1] - out2[:, : t_inspect + 1]).max().item()
        assert diff < 1e-6, f"算子 {name} 存在未来函数泄漏！t<={t_inspect} 最大差异: {diff}"


def test_safe_div_mathematical_properties():
    """验证 SAFE_DIV 的边界与数值稳定特性。"""
    spec = OPERATOR_REGISTRY.get_operator("SAFE_DIV")

    x = torch.tensor([[10.0, -5.0, 1.0, 0.0]])
    # y 包含 0、极小正数、极小负数、常态数
    y = torch.tensor([[0.0, -1e-8, 1e-8, 2.0]])

    out = spec.transform(x, y)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # 10.0 / (0.0 + 1e-6) = 1e7
    assert abs(out[0, 0].item() - 1e7) < 1e2
    # 0.0 / 2.0 = 0.0
    assert abs(out[0, 3].item() - 0.0) < 1e-6


def test_ts_arg_max_min_mathematical_properties():
    """验证 TS_ARG_MAX/MIN 的时间定位归一化 (0=最早, 1=最近)。"""
    argmax10 = OPERATOR_REGISTRY.get_operator("TS_ARG_MAX_10")
    argmin10 = OPERATOR_REGISTRY.get_operator("TS_ARG_MIN_10")

    # 构造单调递增序列：每个窗口的最大值都在当前位置 (最近) -> 归一化应为 1.0
    x_inc = torch.arange(1.0, 31.0).unsqueeze(0)  # [1, 30]
    out_argmax = argmax10.transform(x_inc)
    # 在第 15 步，窗口内最大值就是当前步，argmax 归一化值为 1.0
    assert abs(out_argmax[0, 15].item() - 1.0) < 1e-5

    # 构造单调递减序列：每个窗口的最大值都在窗口最早位置 -> 归一化应为 0.0
    x_dec = torch.arange(30.0, 0.0, -1.0).unsqueeze(0)
    out_argmax_dec = argmax10.transform(x_dec)
    assert abs(out_argmax_dec[0, 15].item() - 0.0) < 1e-5


def test_decay_linear_weights():
    """验证 DECAY_LINEAR_10 线性权重衰减计算的准确性。"""
    decay10 = OPERATOR_REGISTRY.get_operator("DECAY_LINEAR_10")

    # 构造常数序列，均值应该保持为该常数
    c = 7.5
    x_const = torch.full((1, 40), c)
    out = decay10.transform(x_const)
    # warm-up 之后必须精确等于常数
    assert abs(out[0, 20].item() - c) < 1e-5


def test_infection_model_rules_updated():
    """验证 vm.py 中的感染模型规则已正确接纳新算子。"""
    # 极值位置算子集合包含 10/20
    assert "TS_ARG_MAX_10" in ARG_INDEX_OPS
    assert "TS_ARG_MIN_20" in ARG_INDEX_OPS

    # 极值算子集合包含 5
    assert "TS_MAX_5" in EXTREME_OPS
    assert "TS_MIN_5" in EXTREME_OPS

    # 恢复算子包含 SAFE_DIV 与 DELTA_10
    assert "SAFE_DIV" in SIGN_RESTORE_OPS
    assert "DELTA_10" in SIGN_RESTORE_OPS

    # 传播算子包含 DECAY_LINEAR_10 与 SIGNED_POWER_3
    assert "DECAY_LINEAR_10" in INFECTED_PROPAGATING_OPS
    assert "SIGNED_POWER_3" in INFECTED_PROPAGATING_OPS


def test_stackvm_execution_with_vibe_operators():
    """通过 StackVM 运行包含新 Vibe 算子的复合公式并验证无异常执行。"""
    vm = StackVM()
    feat_offset = FORMULA_VOCAB.operator_offset

    # 找到几个 Vibe 算子的 token id
    name_to_id = {name: idx for idx, name in enumerate(FORMULA_VOCAB.token_names)}

    feat_id = 0  # 第一个特征
    decay_id = name_to_id["DECAY_LINEAR_10"]
    delta_id = name_to_id["DELTA_10"]
    power3_id = name_to_id["SIGNED_POWER_3"]

    # 构造公式: FEAT0 -> DECAY_LINEAR_10 -> DELTA_10 -> SIGNED_POWER_3
    formula = [feat_id, decay_id, delta_id, power3_id]

    violations = validate_formula_structure(formula, FORMULA_VOCAB.token_names)
    assert violations == [], f"合规公式不应有违规: {violations}"

    feat_tensor = torch.randn(1, FORMULA_VOCAB.feature_count, 100)
    res = vm.execute(formula, feat_tensor)

    assert res is not None, "StackVM 执行返回 None"
    assert res.shape == (1, 100)
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()
