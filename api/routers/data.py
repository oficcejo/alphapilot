"""
api/routers/data.py — 数据管理路由

数据文件列表、OKX 品种发现、K 线下载。
"""
import pathlib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from data_pipeline.parquet_manager import list_parquet_files, inspect_parquet_file, delete_parquet_file
from data_pipeline.downloader import discover_instruments, discover_swap_instruments, download_symbol_data, download_multi_symbols

router = APIRouter(prefix="/api/data", tags=["数据管理"])


@router.get("/parquets")
async def list_parquets():
    """列出所有本地 Parquet 数据文件。"""
    return {"files": list_parquet_files()}


@router.get("/parquet/{file_name}")
async def get_parquet_info(file_name: str):
    """获取 Parquet 文件详情。"""
    from config import Config
    path = pathlib.Path(Config.DATA_DIR) / file_name
    if not path.exists():
        raise HTTPException(404, f"文件不存在: {file_name}")
    return inspect_parquet_file(str(path))


@router.delete("/parquet/{file_name}")
async def remove_parquet(file_name: str):
    """删除 Parquet 文件。"""
    from config import Config
    path = pathlib.Path(Config.DATA_DIR) / file_name
    if not path.exists():
        raise HTTPException(404, f"文件不存在: {file_name}")
    delete_parquet_file(str(path))
    return {"status": "deleted", "file": file_name}


@router.get("/instruments")
async def get_instruments(inst_type: str = "SWAP"):
    """发现 OKX 可用品种（支持 SPOT/SWAP/FUTURES/OPTION，含 TradFi）。

    Args:
        inst_type: SPOT / SWAP / FUTURES / OPTION
    """
    try:
        instruments = discover_instruments(inst_type)
        return {"instruments": instruments, "count": len(instruments), "inst_type": inst_type}
    except Exception as e:
        raise HTTPException(502, f"OKX API 错误: {e}")


class DownloadRequest(BaseModel):
    symbol: str
    bar: str = "1H"
    total_bars: int = 2000


@router.post("/download")
async def download_data(req: DownloadRequest):
    """下载某品种的 K 线数据。"""
    try:
        result = download_symbol_data(req.symbol, req.bar, req.total_bars)
        return result
    except Exception as e:
        raise HTTPException(502, f"下载失败: {e}")


class DownloadMultiRequest(BaseModel):
    symbols: list[str]
    bar: str = "1H"
    total_bars: int = 2000


@router.post("/download-multi")
async def download_multi(req: DownloadMultiRequest):
    """批量下载多个品种。"""
    try:
        results = download_multi_symbols(req.symbols, req.bar, req.total_bars)
        return {"results": results}
    except Exception as e:
        raise HTTPException(502, f"批量下载失败: {e}")
