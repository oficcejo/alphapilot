"""
config.py — 全局配置（根目录 Config 类）

品种、数据、风控等全局配置统一由此管理。
模型层参数见 model/config.py 的 ModelConfig。
"""
import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STRATEGIES_DIR = BASE_DIR / "strategies"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
WEB_DIR = BASE_DIR / "web"

for _d in (DATA_DIR, STRATEGIES_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Config:
    """全局配置——品种、数据路径、风控、OKX 连接、交易模式。"""

    # ── 路径 ──────────────────────────────────────────────────────────────
    BASE_DIR = str(BASE_DIR)
    DATA_DIR = str(DATA_DIR)
    STRATEGIES_DIR = str(STRATEGIES_DIR)
    CHECKPOINT_DIR = str(CHECKPOINT_DIR)
    WEB_DIR = str(WEB_DIR)
    STRATEGY_FILE = str(pathlib.Path(STRATEGIES_DIR) / "best_mt5_strategy.json")

    # ── OKX API ──────────────────────────────────────────────────────────
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
    OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")
    OKX_API_BASE = os.getenv("OKX_API_BASE", "https://www.okx.com")
    OKX_API_SIMULATED = os.getenv("OKX_SIMULATED", "1") == "1"  # 默认模拟盘
    OKX_BROKER_TAG = "c314b0aecb5bBCDE"

    # ── 交易模式 ──────────────────────────────────────────────────────────
    # 默认 paper（模拟盘），live 需显式开启
    TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # paper | live

    @classmethod
    def is_live(cls) -> bool:
        """是否为实盘模式。需要显式设置 TRADING_MODE=live 且配置完整凭证。"""
        return cls.TRADING_MODE == "live" and bool(
            cls.OKX_API_KEY and cls.OKX_API_SECRET and cls.OKX_API_PASSPHRASE
        )

    @classmethod
    def is_paper(cls) -> bool:
        return not cls.is_live()

    # ── 风控默认值 ────────────────────────────────────────────────────────
    DEFAULT_CAPITAL = 10000.0       # 默认本金 (USDT)
    DEFAULT_LEVERAGE = 5            # 默认杠杆
    MAX_LEVERAGE = 20               # 最大杠杆
    MAX_DAILY_LOSS_PCT = 0.10       # 单日最大亏损 10%
    MAX_POSITION_PCT = 0.30         # 单品种最大仓位占比 30%
    COST_RATE = 0.0005              # 默认手续费率 (0.05%)
    SLIPPAGE = 0.0003               # 默认滑点 (0.03%)

    # ── 默认品种 ──────────────────────────────────────────────────────────
    DEFAULT_SYMBOLS = [
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
    ]

    # ── Web 服务 ──────────────────────────────────────────────────────────
    WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(os.getenv("WEB_PORT", "8009"))


# ── 运行时单例 ─────────────────────────────────────────────────────────────
cfg = Config()
