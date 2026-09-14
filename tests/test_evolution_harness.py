"""
tests/test_evolution_harness.py -- Reef Adaptive Harness 模块单元测试
"""

import json
import shutil
import tempfile
import pathlib
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from config import Config
from strategy_manager.signal import (
    compute_target_positions_stateless,
    LOWER_BAND,
    UPPER_BAND,
)
from evolution.harness import (
    MarketRegime,
    HarnessParams,
    MarketRegimeDetector,
    AdaptiveHarnessPolicy,
    HarnessOptimizer,
)


@pytest.fixture
def temp_evolution_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_harness_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_market_regime_detector_classification():
    detector = MarketRegimeDetector(atr_period=14, baseline_period=50)

    # 1. 模拟平稳数据 (Normal)
    n = 100
    base_price = 2500.0
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 10, size=n)
    close = base_price + np.cumsum(noise)
    high = close + 5.0
    low = close - 5.0

    res = detector.detect_from_arrays(high, low, close)
    assert res["regime"] in [MarketRegime.NORMAL, MarketRegime.VOL_COMPRESSION]
    assert "vol_ratio" in res
    assert "atr_14" in res

    # 2. 模拟波动爆发 (VOL_EXPANSION)
    # 前 80 根 bar 极小波动，最后 20 根 bar 剧烈波动
    close_exp = np.ones(100) * 2500.0
    high_exp = close_exp + 2.0
    low_exp = close_exp - 2.0
    for i in range(80, 100):
        close_exp[i] = 2500.0 + (i - 80) * 25.0
        high_exp[i] = close_exp[i] + 40.0
        low_exp[i] = close_exp[i] - 40.0

    res_exp = detector.detect_from_arrays(high_exp, low_exp, close_exp)
    assert res_exp["regime"] == MarketRegime.VOL_EXPANSION
    assert res_exp["vol_ratio"] >= 1.25 or abs(res_exp["trend_slope"]) >= 0.05

    # 3. 模拟极低波压缩 (VOL_COMPRESSION)
    # 前 80 根 bar 正常波动，最后 20 根 bar 极窄盘整
    close_comp = np.linspace(2500, 2600, 100)
    high_comp = close_comp + 20.0
    low_comp = close_comp - 20.0
    for i in range(80, 100):
        high_comp[i] = close_comp[i] + 1.0
        low_comp[i] = close_comp[i] - 1.0

    res_comp = detector.detect_from_arrays(high_comp, low_comp, close_comp)
    assert res_comp["regime"] == MarketRegime.VOL_COMPRESSION
    assert res_comp["vol_ratio"] <= 0.80


def test_market_regime_insufficient_bars():
    detector = MarketRegimeDetector(atr_period=14, baseline_period=50)
    # 仅 20 根 bar，不足 64
    close = np.array([2500.0] * 20)
    res = detector.detect_from_arrays(close, close, close)
    assert res["regime"] == MarketRegime.NORMAL
    assert res["vol_ratio"] == 1.0


def test_adaptive_harness_policy_evaluate(temp_evolution_dir):
    policy = AdaptiveHarnessPolicy(data_dir=temp_evolution_dir)

    # 构造 mock raw_dict
    n = 100
    close = torch.linspace(2500, 2550, n).unsqueeze(0)
    raw_dict = {
        "close": close,
        "high": close + 10.0,
        "low": close - 10.0,
    }

    params = policy.evaluate(raw_dict)
    assert isinstance(params, HarnessParams)
    assert params.is_dynamic is True
    assert Config.HARNESS_MIN_SL <= params.stop_loss_pct <= Config.HARNESS_MAX_SL
    assert 0.15 <= params.lower_band <= 0.35
    assert 0.60 <= params.upper_band <= 0.85

    # 测试当 ENABLE_DYNAMIC_HARNESS=False 时回退安全基线
    with patch.object(Config, "ENABLE_DYNAMIC_HARNESS", False):
        disabled_params = policy.evaluate(raw_dict)
        assert disabled_params.is_dynamic is False
        assert disabled_params.regime == MarketRegime.NORMAL
        assert disabled_params.stop_loss_pct == Config.HARNESS_DEFAULT_SL
        assert disabled_params.lower_band == 0.25
        assert disabled_params.upper_band == 0.75


def test_compute_target_positions_with_dynamic_bands():
    factors = torch.tensor([[0.20, 0.40, 0.70, 0.90]])

    # 1. 默认参数 (lower_band=0.25, upper_band=0.75)
    default_pos = compute_target_positions_stateless(factors)
    # factors=0.20 -> tanh(0.20)=0.197 < 0.25 -> pos = 0.0
    assert float(default_pos[0, 0].item()) == 0.0
    # factors=0.90 -> tanh(0.90)=0.716 -> between 0.25 and 0.75
    assert float(default_pos[0, 3].item()) > 0.0

    # 2. 压缩体制动态参数 (lower_band=0.18, upper_band=0.60)
    comp_pos = compute_target_positions_stateless(factors, lower_band=0.18, upper_band=0.60)
    # factors=0.20 -> tanh(0.20)=0.197 > 0.18 -> 激活仓位（提前捕捉离场/反转）
    assert float(comp_pos[0, 0].item()) > 0.0

    # 3. 扩张体制动态参数 (lower_band=0.28, upper_band=0.80)
    exp_pos = compute_target_positions_stateless(factors, lower_band=0.28, upper_band=0.80)
    # factors=0.40 -> tanh(0.40)=0.380
    assert float(exp_pos[0, 0].item()) == 0.0


def test_harness_optimizer(temp_evolution_dir):
    optimizer = HarnessOptimizer(data_dir=temp_evolution_dir)

    # 构造含 09-08~09-10 慢阴跌扛单的 mock trajectories
    mock_trajectories = [
        {
            "pos_id": "trade_win_expansion",
            "direction": "long",
            "duration_hours": 20.0,
            "net_pnl": 4.56,
            "pnl_ratio": 0.1173,
        },
        {
            "pos_id": "trade_bleed_compression",
            "direction": "long",
            "duration_hours": 74.2,
            "net_pnl": -10.38,
            "pnl_ratio": -0.157,
        },
    ]

    report = optimizer.evaluate_trajectories(mock_trajectories)
    assert report["total_trades"] == 2
    assert "baseline" in report
    assert "adaptive_harness" in report

    # 验证阴跌痛点单被有效缓解
    comp_trade = [t for t in report["trades_detail"] if t["pos_id"] == "trade_bleed_compression"][0]
    assert comp_trade["mitigated"] is True
    assert comp_trade["pnl_saved"] > 0.0
    assert comp_trade["duration_hours"] <= 24.0
    assert comp_trade["regime_applied"] == MarketRegime.VOL_COMPRESSION

    # 验证总体回撤缩小且总收益提升
    assert report["adaptive_harness"]["total_net_pnl"] > report["baseline"]["total_net_pnl"]
    assert abs(report["adaptive_harness"]["max_drawdown_ratio"]) < abs(report["baseline"]["max_drawdown_ratio"])

    # 验证沉淀配置
    optimizer.auto_tune_and_save()
    assert optimizer.policy.active_file.exists()
