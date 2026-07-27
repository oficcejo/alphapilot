"""
api/main.py — FastAPI 主应用

OKX AlphaPilot | 量化研究与交易中枢

路由：
  /api/data/*       — 数据管理（Parquet 列表、OKX 下载、品种发现）
  /api/training/*   — 模型训练（启动、状态、曲线、断点续训）
  /api/backtest/*   — 策略回测（资金曲线、绩效指标）
  /api/analysis/*   — 实时分析（OKX/MT5/TradingView 信号）
  /api/trading/*    — 实盘交易（paper/live 双模式）

前端：/web/index.html
"""
import pathlib
import sys

# 确保项目根目录在 path 中
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager

from config import Config
from api.routers import data, training, backtest, analysis, trading, portfolio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时打印配置信息与启动 WebSocket。"""
    print("=" * 60)
    print("  OKX AlphaPilot | 量化研究与交易中枢")
    print("=" * 60)
    print(f"  交易模式     : {Config.TRADING_MODE}")
    print(f"  实盘已启用   : {Config.is_live()}")
    print(f"  模拟盘       : {Config.OKX_API_SIMULATED}")
    print(f"  API 已配置   : {bool(Config.OKX_API_KEY)}")
    print(f"  数据目录     : {Config.DATA_DIR}")
    print(f"  策略目录     : {Config.STRATEGIES_DIR}")
    print(f"  检查点目录   : {Config.CHECKPOINT_DIR}")
    print(f"  Web 服务     : http://{Config.WEB_HOST}:{Config.WEB_PORT}")
    print("=" * 60)
    
    # 启动后台 WebSocket 监听循环
    try:
        from data_pipeline.okx_ws_client import okx_ws_client
        import asyncio
        asyncio.create_task(okx_ws_client.start())
        print("  WebSocket 服 : 启动中")
    except Exception as e:
        print(f"  WebSocket 服 : 未启动 ({e})")

    yield


app = FastAPI(
    title="OKX AlphaPilot",
    description="量化研究与交易中枢 — 因子挖掘 · 策略回测 · 实时分析 · 交易执行",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    """禁用前端静态资源 HTTP 缓存，确保 JS/CSS 修改即时生效。"""
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ── 注册路由 ────────────────────────────────────────────────────────────────
app.include_router(data.router)
app.include_router(training.router)
app.include_router(backtest.router)
app.include_router(analysis.router)
app.include_router(trading.router)
app.include_router(portfolio.router)


# ── 系统状态 ────────────────────────────────────────────────────────────────
@app.get("/api/system")
async def system_info():
    """系统信息。"""
    return {
        "name": "OKX AlphaPilot",
        "version": "1.0.0",
        "trading_mode": Config.TRADING_MODE,
        "is_live": Config.is_live(),
        "data_dir": Config.DATA_DIR,
        "strategies_dir": Config.STRATEGIES_DIR,
        "vocab_size": _get_vocab_size(),
    }


def _get_vocab_size() -> int:
    try:
        from model.vocab import FORMULA_VOCAB
        return FORMULA_VOCAB.size
    except Exception:
        return 0


# ── 前端静态文件 ────────────────────────────────────────────────────────────
web_dir = pathlib.Path(Config.WEB_DIR)
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")


@app.get("/")
async def index():
    """返回前端首页。"""
    index_path = web_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "web/index.html not found", "web_dir": str(web_dir)})
