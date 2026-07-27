"""
api/routers/portfolio.py — 多因子组合策略管理路由

支持多因子策略组装、导出与权重调度。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from api.services.strategy_service import create_portfolio_strategy, list_strategies, delete_strategy

router = APIRouter(prefix="/api/portfolio", tags=["组合策略"])


class CreatePortfolioRequest(BaseModel):
    portfolio_name: str
    strategy_paths: List[str]
    weight_method: str = "score_weighted"  # "equal" / "score_weighted" / "manual"
    custom_weights: Optional[List[float]] = None


@router.post("/create")
async def create_portfolio(req: CreatePortfolioRequest):
    """构建多因子组合策略。"""
    try:
        result = create_portfolio_strategy(
            portfolio_name=req.portfolio_name,
            strategy_paths=req.strategy_paths,
            weight_method=req.weight_method,
            custom_weights=req.custom_weights,
        )
        return {"ok": True, "msg": f"组合策略 {result['portfolio_name']} 创建成功", "portfolio": result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"创建组合策略失败: {e}")


@router.get("/list")
async def get_portfolios():
    """获取所有已创建的组合策略列表。"""
    strats = list_strategies()
    portfolios = [s for s in strats if s.get("is_portfolio")]
    return {"portfolios": portfolios}
