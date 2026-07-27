"""
api/routers/backtest.py — 策略回测路由

选择策略和数据、设置手续费/滑点、查看资金曲线。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.backtest_service import backtest_service

router = APIRouter(prefix="/api/backtest", tags=["策略回测"])


class BacktestRequest(BaseModel):
    strategy_path: str
    data_file: str
    cost_rate: float = 0.0005
    slippage: float = 0.0003
    initial_capital: float = 10000.0
    leverage: int = 5


@router.post("/run")
async def run_backtest(req: BacktestRequest):
    """执行回测。"""
    try:
        result = backtest_service.run_backtest(
            strategy_path=req.strategy_path,
            data_file=req.data_file,
            cost_rate=req.cost_rate,
            slippage=req.slippage,
            initial_capital=req.initial_capital,
            leverage=req.leverage,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"回测失败: {e}")
