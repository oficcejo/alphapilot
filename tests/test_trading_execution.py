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
