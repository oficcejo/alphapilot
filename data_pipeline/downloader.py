"""
data_pipeline/downloader.py — OKX K 线数据下载 → Parquet

提供从 OKX 下载历史 K 线并保存为 Parquet 的便捷接口。
"""
import time
from typing import Optional

from .okx_client import OKXClient, get_public_client
from .parquet_manager import save_candles_to_parquet
from .timeframe_utils import parse_bar_to_minutes


def discover_instruments(
    inst_type: str = "SWAP",
    client: OKXClient | None = None,
) -> list[dict]:
    """发现可用品种（支持 SPOT/SWAP/FUTURES/OPTION，含加密、贵金属、指数、股票等 TradFi）。

    Args:
        inst_type: SPOT / SWAP / FUTURES / OPTION
        client: OKX 客户端
    """
    cli = client or get_public_client()
    instruments = cli.get_instruments(inst_type)
    result = []
    for inst in instruments:
        result.append({
            "inst_id": inst.get("instId", ""),
            "inst_type": inst.get("instType", inst_type),
            "settle_ccy": inst.get("settleCcy", ""),
            "ct_val": inst.get("ctVal", ""),
            "ct_val_ccy": inst.get("ctValCcy", ""),
            "state": inst.get("state", ""),
            "lever": inst.get("lever", ""),
            "tick_sz": inst.get("tickSz", ""),
            "lot_sz": inst.get("lotSz", ""),
            "min_sz": inst.get("minSz", ""),
            "base_ccy": inst.get("baseCcy", ""),
            "quote_ccy": inst.get("quoteCcy", ""),
        })
    return result


# 向后兼容
def discover_swap_instruments(client: OKXClient | None = None) -> list[dict]:
    """发现可用 SWAP 合约品种（向后兼容接口）。"""
    return discover_instruments("SWAP", client)


def download_symbol_data(
    symbol: str,
    bar: str = "1H",
    total_bars: int = 2000,
    client: OKXClient | None = None,
) -> dict:
    """下载某品种的 K 线数据并保存为 Parquet。

    Args:
        symbol: 如 BTC-USDT-SWAP
        bar: K线周期
        total_bars: 目标 K 线数量
        client: OKX 客户端

    Returns:
        {"file_path": ..., "n_bars": ..., "symbol": ..., "bar": ...}
    """
    cli = client or get_public_client()
    candles = cli.download_candles(symbol, bar, total_bars)
    if not candles:
        raise RuntimeError(f"未获取到 {symbol} {bar} K线数据")

    path = save_candles_to_parquet(symbol, bar, candles)
    n_bars = len(candles)

    return {
        "file_path": path,
        "n_bars": n_bars,
        "symbol": symbol,
        "bar": bar,
        "status": "ok",
    }


def download_multi_symbols(
    symbols: list[str],
    bar: str = "1H",
    total_bars: int = 2000,
) -> list[dict]:
    """批量下载多个品种。"""
    results = []
    for sym in symbols:
        try:
            r = download_symbol_data(sym, bar, total_bars)
            results.append(r)
        except Exception as e:
            results.append({
                "symbol": sym,
                "bar": bar,
                "status": "error",
                "error": str(e),
            })
        time.sleep(0.3)
    return results
