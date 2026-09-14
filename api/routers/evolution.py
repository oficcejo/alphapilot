"""
api/routers/evolution.py -- Reef 持续自进化中枢路由 (Phase 1: Observe & Alignment)

提供轨迹查询、账单对齐同步、统计与监控接口。
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import numpy as np

from config import Config
from evolution.observer import get_observer
from evolution.harness import get_harness_policy, HarnessOptimizer
from evolution.grow import get_grow_engine
from evolution.shadow import get_shadow_evaluator
from evolution.commit import get_commit_manager
from data_pipeline.okx_client import get_private_client, get_public_client

router = APIRouter(prefix="/api/evolution", tags=["Reef 自进化中枢"])


@router.get("/trajectories")
async def get_trajectories(
    limit: int = Query(50, ge=1, le=500),
    inst_id: Optional[str] = Query(None, description="合约代码过滤，如 ETH-USDT-SWAP"),
):
    """获取已完成对齐的实盘交易轨迹列表。"""
    try:
        observer = get_observer()
        return {
            "status": "success",
            "count": len(observer.get_trajectories(limit=limit, inst_id=inst_id)),
            "trajectories": observer.get_trajectories(limit=limit, inst_id=inst_id),
        }
    except Exception as e:
        raise HTTPException(500, f"获取交易轨迹失败: {e}")


@router.get("/stats")
async def get_stats():
    """获取历史交易轨迹统计指标（胜率、净盈亏、累计手续费、均赢均亏）。"""
    try:
        observer = get_observer()
        return {
            "status": "success",
            "stats": observer.get_summary_stats(),
        }
    except Exception as e:
        raise HTTPException(500, f"获取统计失败: {e}")


@router.post("/sync")
async def trigger_sync(lookback_days: int = Query(14, ge=1, le=90)):
    """手动触发 OKX 账单与实盘持仓因果对齐。"""
    try:
        observer = get_observer()
        res = observer.sync(client=get_private_client(), lookback_days=lookback_days)
        return res
    except Exception as e:
        raise HTTPException(500, f"同步对齐失败: {e}")


@router.get("/status")
async def get_status():
    """获取 Reef 持续自进化中枢（Observe, Harness, Grow, Shadow, Commit）当前运行状态。"""
    try:
        observer = get_observer()
        stats = observer.get_summary_stats()
        policy = get_harness_policy()
        grow = get_grow_engine()
        shadow = get_shadow_evaluator()
        commit_mgr = get_commit_manager()

        return {
            "engine": "Reef Continual Self-Improving Engine",
            "version": "Phase 3 - Full Online Evolution & Shadow Delivery",
            "status": "active",
            "components": {
                "observe": {
                    "total_trajectories": stats.get("total_trades", 0),
                    "win_rate": stats.get("win_rate", 0.0),
                    "total_net_pnl": stats.get("total_net_pnl", 0.0),
                },
                "harness": {
                    "enabled": getattr(Config, "ENABLE_DYNAMIC_HARNESS", True),
                    "min_sl": getattr(Config, "HARNESS_MIN_SL", 0.015),
                    "max_sl": getattr(Config, "HARNESS_MAX_SL", 0.040),
                },
                "grow": {
                    "running": grow.is_running,
                    "status": grow.get_status(),
                },
                "shadow": {
                    "pool_size": len(shadow.get_pool()),
                    "min_improvement": getattr(Config, "SHADOW_MIN_IMPROVEMENT", 0.03),
                },
                "commit": {
                    "auto_commit_enabled": getattr(Config, "AUTO_COMMIT_STRATEGY", False),
                    "total_commits": len(commit_mgr.get_commit_history(limit=100)),
                },
            },
            "data_dir": str(observer.data_dir),
        }
    except Exception as e:
        raise HTTPException(500, f"获取状态失败: {e}")


@router.get("/harness")
async def get_harness_info(
    inst_id: Optional[str] = Query("ETH-USDT-SWAP", description="合约代码"),
    bar: str = Query("1H", description="K线周期"),
):
    """获取当前自适应风控配置与实时体制检测参数。"""
    try:
        policy = get_harness_policy()
        cfg_info = policy.get_active_config()

        live_regime = None
        if inst_id:
            try:
                public_client = get_public_client()
                candles = public_client.get_recent_candles(inst_id, bar=bar, limit=100)
                if candles and len(candles) >= 64:
                    close_arr = np.array([float(c[4]) for c in candles], dtype=np.float64)
                    high_arr = np.array([float(c[2]) for c in candles], dtype=np.float64)
                    low_arr = np.array([float(c[3]) for c in candles], dtype=np.float64)
                    raw_dict = {
                        "close": close_arr,
                        "high": high_arr,
                        "low": low_arr,
                    }
                    live_params = policy.evaluate(raw_dict)
                    live_regime = live_params.to_dict()
            except Exception as ex:
                live_regime = {"error": str(ex)}

        return {
            "status": "success",
            "dynamic_enabled": getattr(Config, "ENABLE_DYNAMIC_HARNESS", True),
            "active_config": cfg_info,
            "live_evaluation": live_regime,
        }
    except Exception as e:
        raise HTTPException(500, f"获取 Harness 配置失败: {e}")


@router.post("/harness/tune")
async def tune_harness():
    """触发基于历史权威实盘轨迹 (trajectories.jsonl) 的离线回放优化并沉淀配置。"""
    try:
        optimizer = HarnessOptimizer()
        result = optimizer.auto_tune_and_save()
        return {
            "status": "success",
            "message": "Reef Harness 自适应寻优完成，已沉淀至 active_harness.json",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(500, f"寻优调优失败: {e}")


# ── Phase 3: Grow, Shadow & Commit ────────────────────────────────────────

class GrowStartRequest(BaseModel):
    strategy_path: str = "strategies/best_ETH-USDT-SWAP_1H.json"
    inst_id: str = "ETH-USDT-SWAP"
    bar: str = "1H"
    steps: int = 100


class CommitRequest(BaseModel):
    candidate_id: str
    target_strategy_path: Optional[str] = None
    force: bool = False


@router.post("/grow/start")
async def start_grow(req: GrowStartRequest):
    """启动后台策略公式进化任务（因果轨迹引导）。"""
    try:
        grow_engine = get_grow_engine()
        res = grow_engine.start_grow(
            strategy_path=req.strategy_path,
            inst_id=req.inst_id,
            bar=req.bar,
            steps=req.steps,
        )
        return res
    except Exception as e:
        raise HTTPException(500, f"启动策略进化失败: {e}")


@router.get("/grow/status")
async def get_grow_status():
    """获取当前策略进化的进度与最佳候选。"""
    try:
        grow_engine = get_grow_engine()
        return {
            "status": "success",
            "grow_status": grow_engine.get_status(),
        }
    except Exception as e:
        raise HTTPException(500, f"获取进化状态失败: {e}")


@router.post("/grow/stop")
async def stop_grow():
    """终止当前策略进化任务。"""
    try:
        grow_engine = get_grow_engine()
        return grow_engine.stop_grow()
    except Exception as e:
        raise HTTPException(500, f"停止进化失败: {e}")


@router.get("/shadow")
async def get_shadow_pool():
    """查看影子观察池中所有候选策略及其 5 重门禁评测报告。"""
    try:
        shadow_evaluator = get_shadow_evaluator()
        candidates = shadow_evaluator.get_pool()
        return {
            "status": "success",
            "count": len(candidates),
            "candidates": candidates,
        }
    except Exception as e:
        raise HTTPException(500, f"获取影子池失败: {e}")


@router.post("/commit")
async def commit_strategy(req: CommitRequest):
    """手动或系统触发将通过 5 重门禁的影子候选策略原子交付至实盘。"""
    try:
        commit_mgr = get_commit_manager()
        res = commit_mgr.commit_candidate(
            candidate_id=req.candidate_id,
            target_strategy_path=req.target_strategy_path,
            force=req.force,
        )
        return res
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, f"策略交付失败: {e}")


@router.get("/commits")
async def get_commit_history(limit: int = Query(20, ge=1, le=100)):
    """获取策略迭代交付与归档审计历史。"""
    try:
        commit_mgr = get_commit_manager()
        commits = commit_mgr.get_commit_history(limit=limit)
        return {
            "status": "success",
            "count": len(commits),
            "commits": commits,
        }
    except Exception as e:
        raise HTTPException(500, f"获取交付历史失败: {e}")


