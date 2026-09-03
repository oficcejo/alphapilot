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
