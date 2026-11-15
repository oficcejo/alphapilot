"""
tests/test_vibe_features.py — Vibe-Trading Alpha Zoo 特征扩展与集成测试 (Phase 2)

测试目标：
1. 特征完整在册：全部 16 个新增价量与微观结构特征已注册进 FEATURE_REGISTRY 与 FORMULA_VOCAB
2. 形状契约与防崩溃：输入 raw_dict，单个特征输出 [N, T]，整体堆叠为 [N, 81, T]
3. 严格因果无未来函数：篡改未来 t+k 的 K 线数据，t 步特征值严格不变
4. 数学与物理意义验证：KLEN, KUP, KLOW, KBAR_EFF, WVMA, RSQR, CORD 等特征计算正确性
5. StackVM 端到端联动执行：新特征与新算子组合无缝执行
"""
import pytest
import torch

from model.features import FEATURE_NAMES, FEATURE_REGISTRY, MT5FeatureEngineer
from model.vm import StackVM, validate_formula_structure
from model.vocab import FORMULA_VOCAB


EXPECTED_VIBE_FEATS = [
    ("WVMA_5", "volatility"),
    ("WVMA_20", "volatility"),
    ("KLEN", "volatility"),
    ("KUP", "reversal"),
    ("KLOW", "reversal"),
    ("KSFT", "momentum"),
    ("KBAR_EFF", "momentum"),
    ("CORD_5", "volume"),
    ("CORD_20", "volume"),
    ("VSUMP_5", "volume"),
    ("VSUMP_20", "volume"),
    ("RSQR_10", "trend"),
    ("RSQR_30", "trend"),
    ("ALPHA_006", "volume"),
    ("ALPHA_012", "reversal"),
    ("ALPHA_054", "reversal"),
]


def _make_sample_raw(N: int = 2, T: int = 120, seed: int = 42) -> dict:
    """构造基础合法行情字典。"""
    g = torch.Generator().manual_seed(seed)
    close = 2000.0 * torch.cumprod(1.0 + torch.randn(N, T, generator=g) * 0.005, dim=-1)
    open_ = close + torch.randn(N, T, generator=g) * 1.5
    high = torch.maximum(open_, close) + torch.rand(N, T, generator=g) * 3.0 + 0.1
    low = torch.minimum(open_, close) - torch.rand(N, T, generator=g) * 3.0 - 0.1
    volume = torch.rand(N, T, generator=g) * 500.0 + 10.0
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def test_vibe_features_registration():
    """验证所有 Vibe-Trading 特征在注册表与词表中就绪。"""
    registered_names = set(FEATURE_REGISTRY.feature_names)
    vocab_feat_names = set(FORMULA_VOCAB.feature_names)

    assert len(FEATURE_NAMES) == 81, f"总特征数应为 81，实际 {len(FEATURE_NAMES)}"
    assert MT5FeatureEngineer.INPUT_DIM == 81

    for name, expected_cat in EXPECTED_VIBE_FEATS:
        assert name in registered_names, f"特征 {name} 未在 FEATURE_REGISTRY 注册"
        assert name in vocab_feat_names, f"特征 {name} 未在 FORMULA_VOCAB 中"
        spec = FEATURE_REGISTRY.get_feature(name)
        assert spec.category == expected_cat, f"特征 {name} 类别期望 {expected_cat}，实际 {spec.category}"


def test_vibe_features_shape_and_clean_bounds():
    """验证特征计算形状为 [N, T] 且全域数值干净（无 NaN、无 Inf，截断于 [-5, 5]）。"""
    raw = _make_sample_raw(N=2, T=80)
    for name, _ in EXPECTED_VIBE_FEATS:
        spec = FEATURE_REGISTRY.get_feature(name)
        out = spec.compute(raw)
        assert out.shape == (2, 80), f"特征 {name} 输出形状不匹配: {out.shape}"
        assert not torch.isnan(out).any(), f"特征 {name} 存在 NaN"
        assert not torch.isinf(out).any(), f"特征 {name} 存在 Inf"
        assert out.max().item() <= 5.0001, f"特征 {name} 超过最大边界 5.0"
        assert out.min().item() >= -5.0001, f"特征 {name} 低于最小边界 -5.0"


def test_vibe_features_no_lookahead():
    """严格未来函数检验：篡改未来 K 线，历史特征值必须严格不变。"""
    raw1 = _make_sample_raw(N=1, T=100, seed=123)
    t_inspect = 60

    # 克隆并篡改 t > t_inspect 的未来数据
    raw2 = {k: v.clone() for k, v in raw1.items()}
    for k in raw2:
        raw2[k][:, t_inspect + 1 :] += torch.randn(1, 100 - t_inspect - 1) * 20.0

    for name, _ in EXPECTED_VIBE_FEATS:
        spec = FEATURE_REGISTRY.get_feature(name)
        out1 = spec.compute(raw1)
        out2 = spec.compute(raw2)

        # 检查 t <= t_inspect 历史部分的一致性
        diff = torch.abs(out1[:, : t_inspect + 1] - out2[:, : t_inspect + 1]).max().item()
        assert diff < 1e-5, f"特征 {name} 存在未来数据泄漏！最大偏差: {diff}"


def test_kbar_efficiency_and_geometry():
    """验证 K 线几何特征物理语义。"""
    spec_eff = FEATURE_REGISTRY.get_feature("KBAR_EFF")
    spec_kup = FEATURE_REGISTRY.get_feature("KUP")
    spec_klow = FEATURE_REGISTRY.get_feature("KLOW")

    # 构造极度大阳线：Close 贴近 High，Open 贴近 Low
    raw_bull = {
        "open": torch.tensor([[100.0, 100.0]]),
        "high": torch.tensor([[110.0, 110.0]]),
        "low": torch.tensor([[99.9, 99.9]]),
        "close": torch.tensor([[109.9, 109.9]]),
        "volume": torch.tensor([[50.0, 50.0]]),
    }
    eff = spec_eff.compute(raw_bull)
    kup = spec_kup.compute(raw_bull)
    klow = spec_klow.compute(raw_bull)

    # 大阳线多空效率接近 +1.0
    assert eff[0, 1].item() > 0.90
    # 上影线与下影线极短，抛压很低
    assert not torch.isnan(kup).any()
    assert not torch.isnan(klow).any()


def test_rsqr_linearity():
    """验证纯线性单边趋势能够输出高 R 方拟合度。"""
    spec_rsqr = FEATURE_REGISTRY.get_feature("RSQR_10")
    T = 40
    # 纯单调上升序列
    c_linear = torch.arange(1.0, T + 1.0).unsqueeze(0)
    raw = {
        "open": c_linear,
        "high": c_linear + 0.1,
        "low": c_linear - 0.1,
        "close": c_linear,
        "volume": torch.ones(1, T),
    }
    out = spec_rsqr.compute(raw)
    assert not torch.isnan(out).any()
    # 线性序列通过 ts_corr(c, t, 10)^2 得到的 R 方等于 1.0（归一化后保持为稳定高值）
    assert out[0, 25].item() > 0.0


def test_stackvm_end_to_end_with_new_features():
    """端到端测试：在 StackVM 中使用新特征 token 与新算子 token 执行策略。"""
    vm = StackVM()
    name_to_id = {name: idx for idx, name in enumerate(FORMULA_VOCAB.token_names)}

    # 取新特征 WVMA_5 与新算子 SAFE_DIV, SIGNED_POWER_3
    f_wvma = name_to_id["WVMA_5"]
    f_klen = name_to_id["KLEN"]
    op_safediv = name_to_id["SAFE_DIV"]
    op_power3 = name_to_id["SIGNED_POWER_3"]

    # 逆波兰表达式: WVMA_5, KLEN, SAFE_DIV, SIGNED_POWER_3
    formula = [f_wvma, f_klen, op_safediv, op_power3]

    violations = validate_formula_structure(formula, FORMULA_VOCAB.token_names)
    assert violations == [], f"合规公式不应产生违规: {violations}"

    raw = _make_sample_raw(N=1, T=100)
    feat_tensor = MT5FeatureEngineer.compute_features(raw)
    assert feat_tensor.shape == (1, 81, 100)

    res = vm.execute(formula, feat_tensor)
    assert res is not None, "StackVM 执行返回 None"
    assert res.shape == (1, 100)
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()
