"""
tests/test_trading_execution.py — 交易执行层单元测试

测试覆盖：
  1. Delta 仓位管理：同向重复信号保持 HOLD（不重复加仓）
  2. Delta 仓位管理：反向信号触发 REVERSE 先平旧再开新
  3. Delta 仓位管理：中性信号触发 CLOSE 平仓
  4. 策略门禁：实盘拒绝运行负分/退化策略
  5. 止损价计算与附带止损逻辑
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from config import Config
from api.services.trading_service import TradingService


@pytest.fixture
def mock_trading_service():
    service = TradingService()
    service._instrument_cache["ETH-USDT-SWAP"] = {
        "ctVal": "0.1",
        "lotSz": "0.01",
        "minSz": "0.01",
        "tickSz": "0.01",
        "ctValCcy": "ETH",
    }
    return service


def test_delta_hold_on_repeated_signal(mock_trading_service):
    """测试当持仓已达到目标时，重复同向信号应返回 HOLD，不下单。"""
    service = mock_trading_service
    
    # 目标仓位计算：capital=100, max_pos=0.3, lev=5, px=2500, ctVal=0.1 -> target_val = 150 * signal
    # 当 signal=-0.5833 时，target_val = -87.5, raw_sz = 87.5 / 250 = 0.35 张 (空仓 -0.35 张)
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "short", "pos": 0.35}
    ]
    mock_client.get_account_summary.return_value = {"total_eq": 100.0, "avail_bal": 80.0}
    
    with patch("api.services.trading_service.get_private_client", return_value=mock_client),          patch("api.services.trading_service.get_public_client", return_value=mock_client),          patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "RET -> NEG", "best_score": 2.5}),          patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)),          patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[-0.5833]])):
        
        # 模拟 800 根 candles (最新价 2500)
        candles = [[str(1600000000000 + i * 900000), "2500", "2510", "2490", "2500", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles
        
        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=100.0,
                leverage=5,
                bar="15m",
                max_position_pct=0.30,
            )
            
            # 验证：当前持仓 -0.35 与目标 -0.35 完全匹配，order 被跳过，状态为 HOLD，未调用 place_order
            assert res["order"]["skipped"] is True
            assert res["order"]["action"] == "HOLD"
            mock_client.place_order.assert_not_called()


def test_delta_close_on_neutral_signal(mock_trading_service):
    """测试当中性信号（平仓）时，已有持仓应触发 CLOSE 平仓。"""
    service = mock_trading_service
    
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "short", "pos": 0.35}
    ]
    mock_client.get_account_summary.return_value = {"total_eq": 100.0, "avail_bal": 80.0}
    mock_client.close_position.return_value = {"code": "0", "msg": "success"}
    
    with patch("api.services.trading_service.get_private_client", return_value=mock_client),          patch("api.services.trading_service.get_public_client", return_value=mock_client),          patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "RET -> NEG", "best_score": 2.5}),          patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)),          patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[0.0]])):
        
        candles = [[str(1600000000000 + i * 900000), "2500", "2510", "2490", "2500", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles
        
        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=100.0,
                leverage=5,
                bar="15m",
            )
            
            # 验证：触发平仓
            assert res["order"]["action"] == "CLOSE"
            mock_client.close_position.assert_called_once()


def test_delta_open_new_position_with_stop_loss(mock_trading_service):
    """测试空仓时开新仓并附带 3% 止损价。"""
    service = mock_trading_service
    
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = []
    mock_client.get_account_summary.return_value = {"total_eq": 100.0, "avail_bal": 80.0}
    mock_client.place_order.return_value = {"clOrdId": "ap123", "tag": "c314b0aecb5bBCDE", "live": True}
    
    with patch("api.services.trading_service.get_private_client", return_value=mock_client),          patch("api.services.trading_service.get_public_client", return_value=mock_client),          patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "RET -> NEG", "best_score": 2.5}),          patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)),          patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[-0.5833]])):
        
        candles = [[str(1600000000000 + i * 900000), "2500", "2510", "2490", "2500", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles
        
        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=100.0,
                leverage=5,
                bar="15m",
            )
            
            # 验证：调用 place_order，且附带了止损价（做空止损为 2500 * 1.03 = 2575.00）
            mock_client.place_order.assert_called_once()
            call_kwargs = mock_client.place_order.call_args.kwargs
            assert call_kwargs["side"] == "sell"
            assert call_kwargs["sl_trigger_px"] == "2575.00"


def test_strategy_quality_gate_blocks_negative_score(mock_trading_service):
    """测试实盘模式下拦截评分 <= 0 的策略。"""
    service = mock_trading_service
    
    with patch("api.services.trading_service.load_strategy", return_value={"formula": [0], "formula_decoded": "RET", "best_score": -1.99}),          patch.object(Config, "TRADING_MODE", "live"):
        
        res = service.execute_signal(
            strategy_path="negative_strategy.json",
            inst_id="ETH-USDT-SWAP",
        )
        
        # 验证：拒绝执行
        assert res["action"] == "拒绝执行"
        assert res["risk_passed"] is False
        assert res["order"]["skipped"] is True


def test_ladder_tp_tier1_trigger(mock_trading_service):
    """测试当纯标的浮盈达到 +2.8% (≥+2.5%) 时，触发 Tier 1 阶梯止盈，仓位上限压至 65% (减仓 35%)。"""
    service = mock_trading_service

    # 当前持有 1.00 张多单，均价 2500
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 1.00, "avg_px": 2500.0}
    ]
    mock_client.get_account_summary.return_value = {"total_eq": 1000.0, "avail_bal": 800.0}
    mock_client.place_order.return_value = {"clOrdId": "ap123", "live": True}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client), \
         patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "TEST", "best_score": 2.5}), \
         patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)), \
         patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[1.0]])):

        # 模拟最新价 2570 (+2.8% 浮盈)
        candles = [[str(1600000000000 + i * 900000), "2500", "2570", "2490", "2570", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles

        with patch.object(Config, "TRADING_MODE", "live"):
            # capital=500, max_pos=0.5, lev=5, px=2570 -> target_value=1250 -> 1250 / 257 = 4.86 -> 限制测试中 raw_target_sz 约为 1.0
            # 这里设置 capital=102.8, max_pos=0.5, lev=5 -> target_val=257 -> raw_sz=1.0 张
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=102.8,
                leverage=5,
                bar="1H",
                max_position_pct=0.50,
                ladder_tp=True,
            )

            # 验证：触发 Tier 1 阶梯止盈
            assert res["ladder_tp"]["enabled"] is True
            assert res["ladder_tp"]["triggered"] is True
            assert res["ladder_tp"]["tier"] == 1
            assert res["ladder_tp"]["cap_ratio"] == 0.65
            assert res["target_held_sz"] == 0.65
            # 减仓 0.35 张卖出
            assert res["delta_sz"] == -0.35
            mock_client.place_order.assert_called_once()
            call_kwargs = mock_client.place_order.call_args.kwargs
            assert call_kwargs["side"] == "sell"
            assert call_kwargs["sz"] == "0.35"
            # 减仓时不额外挂新止损单
            assert call_kwargs.get("sl_trigger_px") is None


def test_ladder_tp_tier2_trigger(mock_trading_service):
    """测试当纯标的浮盈达到 +4.8% (≥+4.5%) 时，触发 Tier 2 阶梯止盈，仓位上限压至 30% (累计减仓 70%)。"""
    service = mock_trading_service

    # 当前持有 1.00 张多单，均价 2500
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 1.00, "avg_px": 2500.0}
    ]
    mock_client.get_account_summary.return_value = {"total_eq": 1000.0, "avail_bal": 800.0}
    mock_client.place_order.return_value = {"clOrdId": "ap123", "live": True}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client), \
         patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "TEST", "best_score": 2.5}), \
         patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)), \
         patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[1.0]])):

        # 模拟最新价 2620 (+4.8% 浮盈)
        candles = [[str(1600000000000 + i * 900000), "2500", "2620", "2490", "2620", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles

        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=104.8,
                leverage=5,
                bar="1H",
                max_position_pct=0.50,
                ladder_tp=True,
            )

            # 验证：触发 Tier 2 阶梯止盈
            assert res["ladder_tp"]["enabled"] is True
            assert res["ladder_tp"]["triggered"] is True
            assert res["ladder_tp"]["tier"] == 2
            assert res["ladder_tp"]["cap_ratio"] == 0.30
            assert res["target_held_sz"] == 0.30
            # 减仓 0.70 张卖出
            assert res["delta_sz"] == -0.70
            mock_client.place_order.assert_called_once()
            call_kwargs = mock_client.place_order.call_args.kwargs
            assert call_kwargs["side"] == "sell"
            assert call_kwargs["sz"] == "0.70"


def test_ladder_tp_disabled_switch(mock_trading_service):
    """测试当 ladder_tp=False 时，即使浮盈达到 +5%，也不触发阶梯止盈，维持原始目标仓位。"""
    service = mock_trading_service

    # 当前持有 1.00 张多单，均价 2500
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 1.00, "avg_px": 2500.0}
    ]
    mock_client.get_account_summary.return_value = {"total_eq": 1000.0, "avail_bal": 800.0}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client), \
         patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "TEST", "best_score": 2.5}), \
         patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)), \
         patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[1.0]])):

        # 模拟最新价 2625 (+5.0% 浮盈)
        candles = [[str(1600000000000 + i * 900000), "2500", "2625", "2490", "2625", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles

        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=105.0,
                leverage=5,
                bar="1H",
                max_position_pct=0.50,
                ladder_tp=False,  # 关闭阶梯止盈
            )

            # 验证：阶梯止盈未启用，未触发
            assert res["ladder_tp"]["enabled"] is False
            assert res["ladder_tp"]["triggered"] is False
            assert res["target_held_sz"] == 1.00
            # 持仓已是 1.00，目标也是 1.00 -> HOLD
            assert res["order"]["skipped"] is True
            assert res["order"]["action"] == "HOLD"
            mock_client.place_order.assert_not_called()


def test_ladder_tp_one_way_ratchet_no_rebuy(mock_trading_service):
    """测试单向棘轮机制：在 Tier 1 减仓后，即使价格回落至 +1.2%，也不触发反向回补加仓。"""
    service = mock_trading_service

    # 模拟历史已触发过 Tier 1 阶梯止盈，当前持有 0.65 张多单
    service._ladder_tp_ratchet["ETH-USDT-SWAP"] = 0.65

    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 0.65, "avg_px": 2500.0}
    ]
    mock_client.get_account_summary.return_value = {"total_eq": 1000.0, "avail_bal": 800.0}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client), \
         patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "TEST", "best_score": 2.5}), \
         patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)), \
         patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[1.0]])):

        # 模拟最新价回落到 2530 (+1.2% 浮盈，低于 2.5%)
        candles = [[str(1600000000000 + i * 900000), "2500", "2530", "2490", "2530", "100", "", "", "1"] for i in range(800)]
        mock_client.get_recent_candles.return_value = candles

        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=101.2,
                leverage=5,
                bar="1H",
                max_position_pct=0.50,
                ladder_tp=True,
            )

            # 验证：单向棘轮有效，虽然当前浮盈仅 1.2%，但仓位上限依旧被锁定在 0.65
            assert res["ladder_tp"]["cap_ratio"] == 0.65
            assert res["target_held_sz"] == 0.65
            # 维持 0.65 持仓，不产生回补加仓！
            assert res["order"]["skipped"] is True
            assert res["order"]["action"] == "HOLD"
            mock_client.place_order.assert_not_called()


def test_runtime_status_extracts_upl_and_margin(mock_trading_service):
    """测试运行状态能正确提取未实现盈亏和保证金占用，不出现 0.00 假零问题。"""
    service = mock_trading_service

    # 模拟 OKX 真实的返回结构：顶层 upl/imr 为空，实际数据在 details 与 positions 中
    mock_client = MagicMock()
    mock_client.get_account_summary.return_value = {
        "total_eq": 114.03,
        "upl": 0.0,  # 顶层因单币种模式返回 0
        "upl_ratio": 0.0,
        "margin": 0.0,
        "margin_ratio": 0.0,
        "avail_bal": 52.87,
        "currency": "USDT",
        "details": [
            {
                "ccy": "USDT",
                "eq": 114.03,
                "cash_bal": 114.85,
                "avail_bal": 52.87,
                "upl": -0.82,
                "imr": 61.98,
            }
        ],
    }
    mock_client.get_positions_detail.return_value = [
        {
            "inst_id": "ETH-USDT-SWAP",
            "pos_side": "short",
            "pos": 1.21,
            "avg_px": 2579.0,
            "last": 2585.0,
            "upl": -0.82,
            "upl_ratio": -0.013,
            "margin": 61.98,
            "lever": 5.0,
        }
    ]

    with patch("api.services.trading_service.get_private_client", return_value=mock_client):
        status = service.get_runtime_status()
        acct = status["account"]
        assert acct is not None
        # 交叉对齐兜底应正确提取真实盈亏与保证金
        assert acct["upl"] == -0.82
        assert acct["margin"] == 61.98
        assert acct["margin_ratio"] > 0.50  # 约 54.35%


def test_ladder_tp_reduction_not_blocked_by_available_balance(mock_trading_service):
    """测试当持仓较重、可用保证金较低时，阶梯减仓/止盈不会被保证金充足性风控误拦截。"""
    service = mock_trading_service
    service._position_cache.clear()
    service._ladder_tp_ratchet.clear()

    # 模拟持有多头 5.83 张 ETH，开仓价 2698.14，最新价 2782.94（浮盈 +3.14%）
    # 账户权益 516 USDT，已用保证金 323 USDT，可用余额仅 193 USDT
    mock_client = MagicMock()
    mock_client.get_account_summary.return_value = {
        "total_eq": 516.0,
        "avail_bal": 193.0,  # 可用余额小于目标仓位总保证金 (约 198.7 USDT)
    }
    mock_client.get_positions_detail.return_value = [
        {
            "inst_id": "ETH-USDT-SWAP",
            "pos_side": "net",
            "pos": 5.83,
            "avg_px": 2698.14,
            "last": 2782.94,
            "margin": 323.0,
            "lever": 5.0,
        }
    ]
    mock_client.get_instrument.return_value = {
        "ctVal": "0.1",
        "lotSz": "0.01",
        "minSz": "0.01",
        "ctValCcy": "ETH",
    }
    mock_client.place_order.return_value = {"ordId": "tp_order_123"}

    candles = [[str(1600000000000 + i * 900000), "2698.14", "2782.94", "2690.0", "2782.94", "100", "", "", "1"] for i in range(800)]
    mock_client.get_recent_candles.return_value = candles

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client), \
         patch("api.services.trading_service.load_strategy", return_value={"formula": [0, 69], "formula_decoded": "TEST", "best_score": 2.5}), \
         patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)), \
         patch("api.services.trading_service.compute_target_positions_stateless", return_value=torch.tensor([[0.80]])):
        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json",
                inst_id="ETH-USDT-SWAP",
                capital=474.96,
                leverage=5,
                bar="1H",
                max_position_pct=0.80,
                ladder_tp=True,
            )

            # 验证风控必须通过（不应被误判为保证金不足）
            assert res["risk_passed"] is True
            # 阶梯 1 必须触发 (限仓 65%)
            assert res["ladder_tp"]["triggered"] is True
            assert res["ladder_tp"]["tier"] == 1
            # 目标张数压缩，应执行卖出减仓
            assert res["delta_sz"] < 0
            mock_client.place_order.assert_called_once()
            call_kwargs = mock_client.place_order.call_args.kwargs
            assert call_kwargs["side"] == "sell"
            assert float(call_kwargs["sz"]) > 0


def test_auto_trade_status_strategy_name(mock_trading_service):
    """测试 get_auto_trade_status 与 get_runtime_status 能正确返回 strategy_name 与 auto_trade 状态。"""
    service = mock_trading_service
    service._auto_trade_state = {
        "running": True,
        "strategy_path": "/var/data/strategies/best_ETH-USDT-SWAP.json",
        "inst_id": "ETH-USDT-SWAP",
        "capital": 1000.0,
        "leverage": 5,
        "bar": "1H",
        "max_position_pct": 0.30,
        "interval_seconds": 3600,
        "ladder_tp": True,
        "started_at": 1700000000,
    }
    status = service.get_auto_trade_status()
    assert status["running"] is True
    assert status["strategy_name"] == "best_ETH-USDT-SWAP.json"
    assert status["inst_id"] == "ETH-USDT-SWAP"

    rt = service.get_runtime_status()
    assert "auto_trade" in rt
    assert rt["auto_trade"]["strategy_name"] == "best_ETH-USDT-SWAP.json"


def test_realtime_ladder_tp_triggers_and_ratchets(mock_trading_service):
    """测试独立高频实时阶梯止盈：
    1. 盘中涨幅达到 Tier 1 (+2.8% >= +2.5%) 触发减仓至 65%；
    2. 价格在 Tier 1 区间微幅波动 (+3.0%) 时单向棘轮防止重复下单；
    3. 价格继续冲高至 Tier 2 (+4.8% >= +4.5%) 触发第二级减仓至 30%；
    4. 价格剧烈冲高回落 (+0.4%) 时，单向棘轮锁定持仓在 30%，绝不反向回补。
    """
    service = mock_trading_service
    service._position_cache.clear()
    service._ladder_tp_ratchet.clear()
    service._ladder_tp_base_sz.clear()

    # 初始持多单 1.0 张，开仓均价 2500.0
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 1.0, "avg_px": 2500.0}
    ]
    mock_client.place_order.return_value = {"ordId": "tp_1", "live": True}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client):

        # ── 阶段 1: 价格上涨至 2570 (+2.8%)，首次触发 Tier 1 ──
        mock_client.get_ticker.return_value = {"last": "2570.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res1 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")

        assert res1 is not None
        assert res1["tier"] == 1
        assert res1["cap_ratio"] == 0.65
        assert res1["target_held_sz"] == 0.65
        assert res1["order_sz"] == 0.35
        assert service._ladder_tp_ratchet["ETH-USDT-SWAP"] == 0.65
        mock_client.place_order.assert_called_once()
        args1 = mock_client.place_order.call_args.kwargs
        assert args1["side"] == "sell"
        assert args1["pos_side"] == "long"
        assert args1["sz"] == "0.35"

        # ── 阶段 2: 持仓已变为 0.65，价格微涨至 2575 (+3.0%)，单向棘轮阻止重复下单 ──
        mock_client.reset_mock()
        mock_client.get_positions_detail.return_value = [
            {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 0.65, "avg_px": 2500.0}
        ]
        mock_client.get_ticker.return_value = {"last": "2575.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res2 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")

        assert res2 is None
        mock_client.place_order.assert_not_called()

        # ── 阶段 3: 价格暴涨至 2620 (+4.8%)，触发 Tier 2 (限仓 30%) ──
        mock_client.reset_mock()
        mock_client.get_ticker.return_value = {"last": "2620.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res3 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")

        assert res3 is not None
        assert res3["tier"] == 2
        assert res3["cap_ratio"] == 0.30
        assert res3["target_held_sz"] == 0.30
        assert res3["order_sz"] == 0.35  # 0.65 - 0.30 = 0.35
        assert service._ladder_tp_ratchet["ETH-USDT-SWAP"] == 0.30
        mock_client.place_order.assert_called_once()
        args3 = mock_client.place_order.call_args.kwargs
        assert args3["side"] == "sell"
        assert args3["sz"] == "0.35"

        # ── 阶段 4: 价格在 2620 见顶后，轻微回撤到 2590 (-1.1% < 2.5%)，不触发止损 ──
        mock_client.reset_mock()
        mock_client.get_positions_detail.return_value = [
            {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 0.30, "avg_px": 2500.0}
        ]
        mock_client.get_ticker.return_value = {"last": "2590.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res4 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")

        assert res4 is None
        mock_client.close_position.assert_not_called()

        # ── 阶段 5: 价格大跳水至 2510，从峰值 2620 回撤超 2.5%，触发高水位追踪止损全清底仓 ──
        mock_client.reset_mock()
        mock_client.get_ticker.return_value = {"last": "2510.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res5 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")

        assert res5 is not None
        assert res5["event"] == "trailing_stop_loss_trigger"
        assert res5["peak_price"] == 2620.0
        mock_client.close_position.assert_called_once_with("ETH-USDT-SWAP", pos_side="long")


def test_realtime_ladder_tp_short_position(mock_trading_service):
    """测试空头仓位的盘中实时阶梯止盈：价格下跌触发买入平空。"""
    service = mock_trading_service
    service._position_cache.clear()
    service._ladder_tp_ratchet.clear()
    service._ladder_tp_base_sz.clear()
    service._peak_price.clear()

    # 初始持空单 1.0 张，开仓均价 2500.0
    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "short", "pos": 1.0, "avg_px": 2500.0}
    ]
    mock_client.place_order.return_value = {"ordId": "tp_short_1", "live": True}
    # 标的价格跌至 2430 (浮盈 (2500-2430)/2500 = +2.8% >= Tier 1)
    mock_client.get_ticker.return_value = {"last": "2430.0"}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client):
        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.check_realtime_ladder_tp("ETH-USDT-SWAP")

        assert res is not None
        assert res["tier"] == 1
        assert res["cap_ratio"] == 0.65
        assert res["target_held_sz"] == -0.65
        assert res["order_sz"] == 0.35
        mock_client.place_order.assert_called_once()
        args = mock_client.place_order.call_args.kwargs
        assert args["side"] == "buy"
        assert args["pos_side"] == "short"
        assert args["sz"] == "0.35"


def test_breakeven_stop_loss_trigger(mock_trading_service):
    """测试保本止损联动机制：
    1. 触发 Tier 1 阶梯止盈后，剩余 65% 持仓；
    2. 价格如果跌回开仓成本线（开仓价 + 0.15%），立即触发保本市价全平，杜绝由赢变输。
    """
    service = mock_trading_service
    service._position_cache.clear()
    service._ladder_tp_ratchet["ETH-USDT-SWAP"] = 0.65
    service._ladder_tp_base_sz["ETH-USDT-SWAP"] = 1.0
    service._peak_price["ETH-USDT-SWAP"] = 2570.0

    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 0.65, "avg_px": 2500.0}
    ]
    mock_client.close_position.return_value = {"code": "0"}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client):

        # ── 价格在 2520 (+0.8%)，高于保本线 2503.75，不触发保本平仓 ──
        mock_client.get_ticker.return_value = {"last": "2520.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res1 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")
        assert res1 is None
        mock_client.close_position.assert_not_called()

        # ── 价格砸破保本线至 2502 (+0.08% <= 2503.75)，触发保本止损 ──
        mock_client.get_ticker.return_value = {"last": "2502.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res2 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")
        assert res2 is not None
        assert res2["event"] == "breakeven_stop_loss_trigger"
        assert res2["breakeven_price"] == 2503.75
        mock_client.close_position.assert_called_once_with("ETH-USDT-SWAP", pos_side="long")


def test_trailing_stop_loss_runner_super_trend(mock_trading_service):
    """测试单边超级大行情下的高水位追踪止损：
    1. 触发 Tier 2 后，持仓锁定在 30% 趋势底仓；
    2. 标的继续暴涨至 3000 (+20% 峰值)；
    3. 行情从 3000 见顶回落 2.67% 到 2920，触发追踪止损，锁定高额利润退出！
    """
    service = mock_trading_service
    service._position_cache.clear()
    service._ladder_tp_ratchet["ETH-USDT-SWAP"] = 0.30
    service._ladder_tp_base_sz["ETH-USDT-SWAP"] = 1.0
    service._peak_price["ETH-USDT-SWAP"] = 2620.0

    mock_client = MagicMock()
    mock_client.get_positions_detail.return_value = [
        {"inst_id": "ETH-USDT-SWAP", "pos_side": "long", "pos": 0.30, "avg_px": 2500.0}
    ]
    mock_client.close_position.return_value = {"code": "0"}

    with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
         patch("api.services.trading_service.get_public_client", return_value=mock_client):

        # 价格飙升至 3000 (+20%)，极值刷新为 3000，当前不回撤，不触发平仓
        mock_client.get_ticker.return_value = {"last": "3000.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res1 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")
        assert res1 is None
        assert service._peak_price["ETH-USDT-SWAP"] == 3000.0
        mock_client.close_position.assert_not_called()

        # 从 3000 回落至 2920（回撤 2.67% >= 2.5%），触发追踪止损全平底仓
        mock_client.get_ticker.return_value = {"last": "2920.0"}
        with patch.object(Config, "TRADING_MODE", "live"):
            res2 = service.check_realtime_ladder_tp("ETH-USDT-SWAP")
        assert res2 is not None
        assert res2["event"] == "trailing_stop_loss_trigger"
        assert res2["peak_price"] == 3000.0
        assert res2["trailing_sl_px"] == 3000.0 * (1.0 - 0.025)
        mock_client.close_position.assert_called_once_with("ETH-USDT-SWAP", pos_side="long")




