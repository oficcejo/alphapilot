"""
tests/test_evolution_observer.py -- Reef Observe 模块单元测试与断言
"""

import json
import pytest
import shutil
import tempfile
import pathlib
from unittest.mock import MagicMock

from evolution.observer import ObserveEngine


@pytest.fixture
def temp_evolution_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_evo_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_record_receipt(temp_evolution_dir):
    engine = ObserveEngine(data_dir=temp_evolution_dir)
    receipt_id = engine.record_receipt(
        inst_id="ETH-USDT-SWAP",
        side="buy",
        pos_side="long",
        signal=0.75,
        action="做多 75.0%",
        last_price=2500.0,
        target_sz=1.0,
        delta_sz=1.0,
        strategy_formula="TEST_FORMULA -> SIGN",
        bar="15m",
        cl_ord_id="ap_test_123",
    )

    assert receipt_id.startswith("rcpt_")
    assert receipt_id in engine._receipt_cache
    assert engine.receipts_file.exists()

    records = [json.loads(l) for l in engine.receipts_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(records) == 1
    assert records[0]["receipt_id"] == receipt_id
    assert records[0]["strategy"] == "TEST_FORMULA -> SIGN"
    assert records[0]["signal"] == 0.75


def test_observe_sync_and_alignment(temp_evolution_dir):
    engine = ObserveEngine(data_dir=temp_evolution_dir)

    # 1. 模拟开仓录入 Receipt
    engine.record_receipt(
        inst_id="ETH-USDT-SWAP",
        side="buy",
        pos_side="long",
        signal=0.8,
        action="做多 80%",
        last_price=2400.0,
        target_sz=1.0,
        delta_sz=1.0,
        strategy_formula="FORMULA_A -> MAX",
        bar="15m",
        cl_ord_id="ap_test_456",
    )

    # 2. Mock OKX 客户端历史平仓与账单
    mock_client = MagicMock()
    mock_pos_history = [
        {
            "posId": "mock_pos_1",
            "instId": "ETH-USDT-SWAP",
            "direction": "long",
            "openAvgPx": "2400.0",
            "closeAvgPx": "2500.0",
            "pnl": "10.0",
            "pnlRatio": "0.2083",
            "type": "2",
            "cTime": "1789100000000",
            "uTime": "1789110000000",
        },
        {
            "posId": "mock_pos_2",
            "instId": "ETH-USDT-SWAP",
            "direction": "long",
            "openAvgPx": "2500.0",
            "closeAvgPx": "2450.0",
            "pnl": "-5.0",
            "pnlRatio": "-0.10",
            "type": "2",
            "cTime": "1789120000000",
            "uTime": "1789130000000",
        }
    ]
    mock_bills = [
        {"ts": "1789100005000", "type": "2", "fee": "-0.05", "pnl": "0"},
        {"ts": "1789105000000", "type": "8", "fee": "0", "pnl": "-0.01"},
        {"ts": "1789110005000", "type": "2", "fee": "-0.05", "pnl": "10.0"},
        {"ts": "1789120005000", "type": "2", "fee": "-0.04", "pnl": "0"},
        {"ts": "1789130005000", "type": "2", "fee": "-0.04", "pnl": "-5.0"},
    ]
    mock_client.get_positions_history.return_value = mock_pos_history
    mock_client.get_bills.return_value = mock_bills

    # 执行同步
    res = engine.sync(client=mock_client, lookback_days=30)
    assert res["status"] == "success"
    assert res["new_aligned_count"] == 2
    assert res["total_trajectories"] == 2

    # 验证轨迹计算
    trajs = engine.get_trajectories()
    assert len(trajs) == 2

    t1 = next(t for t in trajs if t["open_avg_px"] == 2400.0)
    assert t1["gross_pnl"] == 10.0
    assert abs(t1["total_fee"] - (-0.10)) < 1e-4
    assert abs(t1["total_funding"] - (-0.01)) < 1e-4
    assert abs(t1["net_pnl"] - 9.89) < 1e-4
    assert t1["feedback_score"] > 0.9

    t2 = next(t for t in trajs if t["open_avg_px"] == 2500.0)
    assert t2["gross_pnl"] == -5.0
    assert abs(t2["total_fee"] - (-0.08)) < 1e-4
    assert abs(t2["net_pnl"] - (-5.08)) < 1e-4
    assert t2["feedback_score"] < 0

    # 验证幂等性
    res_idempotent = engine.sync(client=mock_client, lookback_days=30)
    assert res_idempotent["new_aligned_count"] == 0
    assert len(engine.get_trajectories()) == 2

    # 验证统计指标
    stats = engine.get_summary_stats()
    assert stats["total_trades"] == 2
    assert stats["win_count"] == 1
    assert stats["loss_count"] == 1
    assert stats["win_rate"] == 0.5
    assert abs(stats["total_net_pnl"] - (9.89 - 5.08)) < 1e-4
