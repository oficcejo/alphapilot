"""
api/routers/training.py — 模型训练路由

启动训练、停止训练、查看状态、查看训练曲线、断点续训、导出策略。
"""
import pathlib
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from api.services.training_service import training_service
from api.services.strategy_service import list_strategies, load_strategy, delete_strategy, import_strategy

router = APIRouter(prefix="/api/training", tags=["模型训练"])


class TrainRequest(BaseModel):
    data_file: str
    symbol: Optional[str] = None
    train_steps: int = 0
    reward_mode: str = "ftmo"  # ftmo / standard / forex


class ResetRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    delete_strategy: bool = False
    delete_history: bool = True


@router.post("/start")
async def start_training(req: TrainRequest):
    """启动训练任务。"""
    import pathlib
    if not pathlib.Path(req.data_file).exists():
        raise HTTPException(404, f"数据文件不存在: {req.data_file}")
    return training_service.start_training(
        data_file=req.data_file,
        symbol=req.symbol,
        train_steps=req.train_steps,
        reward_mode=req.reward_mode,
    )


@router.get("/status")
async def get_status():
    """获取当前训练状态。"""
    return training_service.status


@router.post("/stop")
async def stop_training():
    """停止当前训练任务（优雅退出，保存当前最优策略）。"""
    return training_service.stop_training()


@router.post("/reset")
async def reset_training(req: ResetRequest):
    """重置训练状态。

    清除指定品种/周期的策略、历史，使下一次训练从头开始。
    - symbol + timeframe 都不传：清除所有品种所有周期
    - 只传 symbol：清除该品种所有周期
    - 传 symbol + timeframe：只清除该品种该周期
    """
    return training_service.reset_training(
        symbol=req.symbol,
        timeframe=req.timeframe,
        delete_strategy=req.delete_strategy,
        delete_history=req.delete_history,
    )


@router.get("/export")
async def export_strategy(strategy_path: str):
    """导出策略 JSON 文件（下载）。

    Args:
        strategy_path: 策略文件路径
    """
    p = pathlib.Path(strategy_path)
    if not p.exists():
        raise HTTPException(404, f"策略文件不存在: {strategy_path}")
    # 验证文件可读
    try:
        load_strategy(strategy_path)
    except Exception as e:
        raise HTTPException(400, f"策略文件无效: {e}")
    return FileResponse(
        path=str(p),
        filename=p.name,
        media_type="application/json",
    )


@router.get("/history")
async def get_history(symbol: Optional[str] = None, timeframe: Optional[str] = None):
    """获取训练历史曲线。

    Args:
        symbol: 品种标识
        timeframe: K 线周期（如 '1H', '5m'）
    """
    return training_service.get_training_history(symbol, timeframe)


@router.get("/strategies")
async def get_strategies():
    """列出已训练的策略。"""
    return {"strategies": list_strategies()}


class DeleteStrategyRequest(BaseModel):
    strategy_path: str


@router.post("/strategies/delete")
async def delete_strategy_route(req: DeleteStrategyRequest):
    """删除指定策略 JSON 文件。

    安全校验：文件必须位于 strategies 目录内。
    """
    try:
        return delete_strategy(req.strategy_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"删除策略失败: {e}")


@router.post("/strategies/import")
async def import_strategy_route(file: UploadFile = File(...)):
    """导入策略 JSON 文件。"""
    try:
        contents = await file.read()
        result = import_strategy(contents, file.filename)
        return {"ok": True, "msg": f"策略 {result['file_name']} 导入成功", "strategy": result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"导入策略失败: {e}")

