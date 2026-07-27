"""
api/routers/trading.py — 实盘交易路由

paper / live 双模式交易执行。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.trading_service import trading_service
from config import Config

router = APIRouter(prefix="/api/trading", tags=["实盘交易"])


class TradeRequest(BaseModel):
    strategy_path: str
    inst_id: str
    capital: float = Config.DEFAULT_CAPITAL
    leverage: int = Config.DEFAULT_LEVERAGE
    bar: str = "1H"
    max_position_pct: float = Config.MAX_POSITION_PCT


@router.get("/status")
async def get_status():
    """获取交易服务状态。"""
    return trading_service.get_status()


@router.get("/runtime")
async def get_runtime_status():
    """获取运行时状态：账户余额、持仓、单日风控、审计统计。

    供前端「运行状态」面板实时刷新。
    """
    try:
        return trading_service.get_runtime_status()
    except Exception as e:
        raise HTTPException(500, f"获取运行状态失败: {e}")


@router.post("/execute")
async def execute_signal(req: TradeRequest):
    """执行交易信号。"""
    try:
        return trading_service.execute_signal(
            strategy_path=req.strategy_path,
            inst_id=req.inst_id,
            capital=req.capital,
            leverage=req.leverage,
            bar=req.bar,
            max_position_pct=req.max_position_pct,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"交易执行失败: {e}")


@router.post("/close/{inst_id}")
async def close_position(inst_id: str):
    """平仓。"""
    try:
        return trading_service.close_position(inst_id)
    except Exception as e:
        raise HTTPException(500, f"平仓失败: {e}")


class AutoTradeRequest(BaseModel):
    strategy_path: str
    inst_id: str
    capital: float = Config.DEFAULT_CAPITAL
    leverage: int = Config.DEFAULT_LEVERAGE
    bar: str = "1H"
    max_position_pct: float = Config.MAX_POSITION_PCT
    interval_seconds: int = 3600


@router.post("/auto/start")
async def start_auto_trade(req: AutoTradeRequest):
    """启动自动交易——按固定间隔循环执行信号。"""
    try:
        return trading_service.start_auto_trade(
            strategy_path=req.strategy_path,
            inst_id=req.inst_id,
            capital=req.capital,
            leverage=req.leverage,
            bar=req.bar,
            max_position_pct=req.max_position_pct,
            interval_seconds=req.interval_seconds,
        )
    except Exception as e:
        raise HTTPException(500, f"启动自动交易失败: {e}")


@router.post("/auto/stop")
async def stop_auto_trade():
    """停止自动交易。"""
    try:
        return trading_service.stop_auto_trade()
    except Exception as e:
        raise HTTPException(500, f"停止自动交易失败: {e}")


@router.get("/auto/status")
async def get_auto_trade_status():
    """获取自动交易状态。"""
    return trading_service.get_auto_trade_status()


@router.get("/audit")
async def get_audit_log(n: int = 50):
    """获取审计日志。"""
    return {"logs": trading_service.get_audit_log(n)}


@router.get("/config")
async def get_trading_config():
    """获取交易配置。"""
    return {
        "mode": Config.TRADING_MODE,
        "is_live": Config.is_live(),
        "is_paper": Config.is_paper(),
        "api_configured": bool(
            Config.OKX_API_KEY and Config.OKX_API_SECRET and Config.OKX_API_PASSPHRASE
        ),
        "simulated": Config.OKX_API_SIMULATED,
        "default_capital": Config.DEFAULT_CAPITAL,
        "default_leverage": Config.DEFAULT_LEVERAGE,
        "max_leverage": Config.MAX_LEVERAGE,
        "max_daily_loss_pct": Config.MAX_DAILY_LOSS_PCT,
        "max_position_pct": Config.MAX_POSITION_PCT,
        "cost_rate": Config.COST_RATE,
        "slippage": Config.SLIPPAGE,
        "total_cost_rate": Config.COST_RATE + Config.SLIPPAGE,
    }
