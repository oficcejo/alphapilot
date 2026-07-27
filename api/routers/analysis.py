"""
api/routers/analysis.py — 实时分析路由

选择 OKX / MT5 / TradingView 数据源，计算最新信号。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.services.analysis_service import analysis_service

router = APIRouter(prefix="/api/analysis", tags=["实时分析"])


class OkxAnalysisRequest(BaseModel):
    strategy_path: str
    inst_id: str
    bar: str = "1H"
    limit: int = 300


class ParquetAnalysisRequest(BaseModel):
    strategy_path: str
    data_file: str


@router.post("/okx")
async def analyze_okx(req: OkxAnalysisRequest):
    """从 OKX 获取实时行情并计算信号。"""
    try:
        return analysis_service.analyze_okx(
            strategy_path=req.strategy_path,
            inst_id=req.inst_id,
            bar=req.bar,
            limit=req.limit,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"OKX 分析失败: {e}")


@router.post("/parquet")
async def analyze_parquet(req: ParquetAnalysisRequest):
    """从本地 Parquet 分析信号（MT5 / TradingView 模式）。"""
    try:
        return analysis_service.analyze_parquet(
            strategy_path=req.strategy_path,
            data_file=req.data_file,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"分析失败: {e}")
