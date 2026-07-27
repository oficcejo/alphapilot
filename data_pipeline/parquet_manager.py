"""
data_pipeline/parquet_manager.py — Parquet 数据文件管理

负责从 OKX 下载 K 线、保存为 Parquet、读取为模型可用的 raw_dict。
"""
import json
import pathlib
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import Config
from .timeframe_utils import parse_bar_to_minutes


def _data_dir() -> pathlib.Path:
    d = pathlib.Path(Config.DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_parquet_files() -> list[dict]:
    """列出 data/ 目录下所有 Parquet 文件及元信息。"""
    files = sorted(_data_dir().glob("*.parquet"))
    result = []
    for f in files:
        info = inspect_parquet_file(str(f))
        info["file_name"] = f.name
        info["file_path"] = str(f)
        info["file_size_mb"] = round(f.stat().st_size / 1024 / 1024, 2)
        result.append(info)
    return result


def inspect_parquet_file(path: str) -> dict:
    """读取 Parquet 文件元信息（不加载全部数据）。"""
    try:
        df = pd.read_parquet(path, columns=None)
        n_bars = len(df)
        symbol = "unknown"
        timeframe = "1H"
        start_ts = ""
        end_ts = ""

        # 推断 symbol / timeframe
        p = pathlib.Path(path)
        name_parts = p.stem.split("_")
        if len(name_parts) >= 2:
            symbol = name_parts[0]
            timeframe = name_parts[1]

        if "time" in df.columns:
            t = df["time"]
            if t.dtype.kind in ('i', 'u', 'f'):
                start_ts = str(int(t.iloc[0]))
                end_ts = str(int(t.iloc[-1]))
            else:
                start_ts = str(t.iloc[0])
                end_ts = str(t.iloc[-1])

        cols = list(df.columns)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "n_bars": n_bars,
            "columns": cols,
            "start_time": start_ts,
            "end_time": end_ts,
        }
    except Exception as e:
        return {"error": str(e), "symbol": "unknown", "timeframe": "unknown",
                "n_bars": 0, "columns": [], "start_time": "", "end_time": ""}


def load_parquet_to_raw_dict(path: str) -> dict:
    """加载 Parquet 为模型可用的 raw_dict。

    Returns:
        dict with keys: close, open, high, low, volume, time (numpy arrays [N, T])
    """
    df = pd.read_parquet(path)
    # 确保列存在
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Parquet 缺少必需列: {col}")

    # 填充 NaN（pandas 3.0 兼容：fillna(method=) 已移除，改用 .ffill()）
    df = df.ffill().fillna(0)

    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # time 列
    if "time" in df.columns:
        t = df["time"].values
        if t.dtype.kind in ('i', 'u', 'f'):
            time_arr = t.astype(np.float64)
        else:
            time_arr = np.arange(len(df), dtype=np.float64)
    else:
        time_arr = np.arange(len(df), dtype=np.float64)

    # 转为 [N=1, T] 张量（单品种）
    raw_dict = {
        "close": torch.from_numpy(close).unsqueeze(0).float(),
        "open": torch.from_numpy(open_).unsqueeze(0).float(),
        "high": torch.from_numpy(high).unsqueeze(0).float(),
        "low": torch.from_numpy(low).unsqueeze(0).float(),
        "volume": torch.from_numpy(volume).unsqueeze(0).float(),
        "time": time_arr,
    }
    return raw_dict


def save_candles_to_parquet(
    symbol: str,
    bar: str,
    candles: list[list],
    output_dir: Optional[str] = None,
) -> str:
    """将 OKX K 线数据保存为 Parquet。

    Args:
        symbol: 如 BTC-USDT-SWAP
        bar: 如 1H
        candles: OKX 返回的 [ts, o, h, l, c, vol, ...] 列表
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    if not candles:
        raise ValueError("candles 为空")

    out_dir = pathlib.Path(output_dir) if output_dir else _data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in candles:
        # OKX K线: [ts, o, h, l, c, vol, volCcy, volQuote, confirm]
        if len(c) >= 6:
            rows.append({
                "time": int(c[0]) // 1000 if int(c[0]) > 1e12 else int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)

    fname = f"{symbol}_{bar}.parquet"
    path = out_dir / fname
    df.to_parquet(path, engine="pyarrow", index=False)

    meta = {
        "symbol": symbol,
        "timeframe": bar,
        "n_bars": len(df),
        "file_path": str(path),
    }
    # 保存元信息 json
    meta_path = out_dir / f"{symbol}_{bar}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(path)


def delete_parquet_file(path: str) -> bool:
    """删除 Parquet 文件及其元信息。"""
    p = pathlib.Path(path)
    if p.exists():
        p.unlink()
    meta = p.with_suffix(".meta.json")
    if meta.exists():
        meta.unlink()
    return True


def get_data_info_for_symbol(symbol: str) -> dict | None:
    """查找某品种的数据文件。"""
    files = list_parquet_files()
    for f in files:
        if f.get("symbol") == symbol:
            return f
    return None
