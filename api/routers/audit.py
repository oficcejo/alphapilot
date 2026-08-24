"""
api/routers/audit.py — 策略审计与防泄漏自检路由

  POST /api/audit/run       独立严谨回测审计（下一开盘成交口径）
  POST /api/audit/selftest  未来函数防泄漏回归测试
"""
import json
import pathlib
import threading
import time
import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import Config
from rigorous_backtest_audit import audit_strategy, COST_RATE, SLIPPAGE

router = APIRouter(prefix="/api/audit", tags=["策略审计"])

# torch 重计算任务串行化，避免并发请求打爆内存
_audit_lock = threading.Lock()


class AuditRequest(BaseModel):
    strategy_path: str | None = None   # None = 审计 strategies/ 全部
    data_file: str | None = None       # None = 按策略元数据自动定位
    leverage: int = 1
    cost_rate: float = COST_RATE
    slippage: float = SLIPPAGE


@router.post("/run")
async def run_audit(req: AuditRequest):
    """执行严谨回测审计：全样本 / 前后半段 / 尾段20% / 成本压力。"""
    if req.strategy_path:
        paths = [req.strategy_path]
    else:
        sdir = pathlib.Path(Config.STRATEGIES_DIR)
        paths = sorted(str(p) for p in sdir.glob("*.json"))
    if not paths:
        raise HTTPException(404, "未找到任何策略文件")

    reports = []
    with _audit_lock:
        for p in paths:
            try:
                r = audit_strategy(
                    p, req.data_file,
                    leverage=req.leverage,
                    cost_rate=req.cost_rate,
                    slippage=req.slippage,
                )
            except Exception as e:
                r = {
                    "strategy": pathlib.Path(p).name,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=3),
                }
            reports.append(r)

    saved_to = None
    try:
        out = pathlib.Path("rigorous_audit_report.json")
        out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_to = out.name
    except Exception:
        pass

    return {"reports": reports, "saved_to": saved_to}


@router.post("/selftest")
async def run_selftest():
    """运行未来函数防泄漏回归测试（tests/test_no_lookahead.py）。"""
    results = []
    with _audit_lock:
        try:
            import tests.test_no_lookahead as m
        except Exception as e:
            raise HTTPException(500, f"无法加载测试模块: {e}")

        test_names = [n for n in dir(m) if n.startswith("test_")]
        for name in test_names:
            fn = getattr(m, name)
            if not callable(fn):
                continue
            t0 = time.perf_counter()
            try:
                fn()
                passed, err = True, None
            except AssertionError as e:
                passed, err = False, str(e) or "断言失败"
            except Exception as e:
                passed, err = False, f"{type(e).__name__}: {e}"
            results.append({
                "name": name,
                "passed": passed,
                "error": err,
                "ms": round((time.perf_counter() - t0) * 1000),
            })

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "tests": results,
        "passed_count": passed_count,
        "total": len(results),
        "all_passed": passed_count == len(results),
    }
