"""
api/services/training_service.py — 训练编排服务

在后台线程中运行 AlphaEngine 训练，支持断点续训。
"""
import sys
import json
import pathlib
import threading
import traceback
from typing import Optional

import torch
import numpy as np

from config import Config
from model.config import ModelConfig
from model.engine import AlphaEngine
from data_pipeline.parquet_manager import load_parquet_to_raw_dict, inspect_parquet_file
from data_pipeline.timeframe_utils import infer_periods_per_year
from model.features import MT5FeatureEngineer


class DataManager:
    """轻量数据管理器——从 Parquet 加载特征和目标收益。"""

    def __init__(self, raw_dict: dict):
        self.raw_dict = raw_dict
        self.feat_tensor = MT5FeatureEngineer.compute_features(raw_dict)
        # target_ret: 下一 bar 的对数收益
        close = raw_dict["close"]  # [N, T]
        eps = 1e-9
        log_ret = torch.zeros_like(close)
        log_ret[:, 1:] = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        self.target_ret = log_ret


class TrainingService:
    """训练服务——管理后台训练任务。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._task: dict | None = None
        self._thread: threading.Thread | None = None
        self._engine: AlphaEngine | None = None   # 指向当前训练引擎，用于停止

    @property
    def status(self) -> dict:
        with self._lock:
            if self._task is None:
                return {"running": False, "message": "无训练任务"}
            task = dict(self._task)  # shallow copy

        # 训练进行中时，从实时历史文件读取当前进度
        if task.get("running") and not task.get("error"):
            try:
                hist_path = task.get("history_path", "")
                p = pathlib.Path(hist_path)
                if p.exists():
                    hist = json.loads(p.read_text(encoding="utf-8"))
                    steps = hist.get("step", [])
                    best_scores = hist.get("best_score", [])
                    if steps:
                        # step 是 0-based，current_step 显示为已完成步数
                        task["current_step"] = steps[-1] + 1
                    if best_scores:
                        task["best_score"] = best_scores[-1]
                    # 读取更多实时字段（如果历史中有）
                    if "restart_count" in hist:
                        task["restart_count"] = hist["restart_count"]
            except Exception:
                pass

        return {
            "running": task.get("running", False),
            "symbol": task.get("symbol"),
            "timeframe": task.get("timeframe"),
            "data_file": task.get("data_file"),
            "start_step": task.get("start_step", 0),
            "end_step": task.get("end_step", 0),
            "current_step": task.get("current_step", 0),
            "best_score": task.get("best_score", -999),
            "best_formula": task.get("best_formula"),
            "best_formula_decoded": task.get("best_formula_decoded"),
            "error": task.get("error"),
            "restart_count": task.get("restart_count", 0),
            "history_path": task.get("history_path"),
            "strategy_path": task.get("strategy_path"),
        }

    def start_training(
        self,
        data_file: str,
        symbol: Optional[str] = None,
        train_steps: int = 0,
        reward_mode: str = "ftmo",
    ) -> dict:
        """启动训练任务。

        Args:
            data_file: Parquet 文件路径
            symbol: 品种标识
            train_steps: 训练步数（0=用默认）
            reward_mode: ftmo / standard / forex

        Returns:
            {"status": "started", ...}
        """
        with self._lock:
            if self._task and self._task.get("running"):
                return {"status": "already_running", "message": "已有训练任务在运行"}

        # 设置 reward mode
        ModelConfig.REWARD_MODE = reward_mode
        steps = train_steps or ModelConfig.TRAIN_STEPS

        # 从数据文件推断 timeframe
        timeframe = None
        try:
            info = inspect_parquet_file(data_file)
            timeframe = info.get("timeframe")
            # 如果未显式传入 symbol，从数据文件推断
            if not symbol:
                symbol = info.get("symbol")
        except Exception:
            pass

        # 构造策略文件名和检查点标记
        parts = [symbol] if symbol else []
        if timeframe:
            parts.append(timeframe)
        tag = "_".join(parts) if parts else ""
        strategy_name = f"best_{tag}.json" if tag else pathlib.Path(Config.STRATEGY_FILE).name
        strategy_path = str(pathlib.Path(Config.STRATEGIES_DIR) / strategy_name)
        history_name = f"training_history_{tag}.json" if tag else "training_history.json"
        history_path = history_name  # 训练历史在项目根目录

        self._task = {
            "running": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data_file": data_file,
            "start_step": 0,
            "end_step": steps,
            "current_step": 0,
            "best_score": -999,
            "best_formula": None,
            "best_formula_decoded": None,
            "error": None,
            "restart_count": 0,
            "history_path": history_path,
            "strategy_path": strategy_path,
        }

        self._thread = threading.Thread(
            target=self._run_training,
            args=(data_file, symbol, timeframe, steps),
            daemon=True,
        )
        self._thread.start()
        return {"status": "started", "symbol": symbol, "timeframe": timeframe, "steps": steps}

    def stop_training(self) -> dict:
        """请求停止当前训练任务。

        通过设置 engine.stop_requested = True，训练循环在下一次迭代时
        优雅退出，保存当前最优策略。
        """
        with self._lock:
            if not self._task or not self._task.get("running"):
                return {"status": "not_running", "message": "无训练任务在运行"}
            engine = self._engine

        if engine is not None:
            engine.stop_requested = True
            return {"status": "stopping", "message": "已请求停止，训练将在当前步完成后退出"}
        return {"status": "error", "message": "引擎实例不可用"}

    def _run_training(self, data_file: str, symbol: Optional[str], timeframe: Optional[str],
                      steps: int):
        """训练线程主函数。"""
        try:
            # 加载数据
            raw_dict = load_parquet_to_raw_dict(data_file)
            # 保留文件路径供引擎写入策略元数据
            if isinstance(raw_dict, dict):
                raw_dict["file_path"] = data_file
            dm = DataManager(raw_dict)

            # 创建引擎（传入 timeframe 和 data_file）
            engine = AlphaEngine(
                data_manager=dm,
                target_symbol=symbol,
                timeframe=timeframe,
                data_file=data_file,
            )
            with self._lock:
                self._engine = engine

            # 训练
            engine.train(start_step=0, end_step=steps, verbose_header=True)

            stopped = engine.stop_requested
            with self._lock:
                if self._task:
                    self._task["running"] = False
                    self._task["current_step"] = steps
                    self._task["best_score"] = engine.best_score
                    self._task["best_formula"] = engine.best_formula
                    self._task["best_formula_decoded"] = engine._decode_formula(engine.best_formula)
                    self._task["restart_count"] = engine._restart_count
                    self._task["stopped"] = stopped
                    self._engine = None

        except Exception as e:
            tb = traceback.format_exc()
            with self._lock:
                if self._task:
                    self._task["running"] = False
                    self._task["error"] = str(e)
                    self._task["traceback"] = tb
                    self._engine = None

    def get_training_history(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> dict:
        """读取训练历史 JSON。"""
        parts = []
        if symbol:
            parts.append(symbol)
        if timeframe:
            parts.append(timeframe)
        sym_tag = f"_{'_'.join(parts)}" if parts else ""
        hist_path = f"training_history{sym_tag}.json"
        p = pathlib.Path(hist_path)
        if not p.exists():
            return {"error": "训练历史文件不存在", "path": str(p)}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data
        except Exception as e:
            return {"error": str(e)}

    def reset_training(self, symbol: Optional[str] = None, timeframe: Optional[str] = None,
                       delete_strategy: bool = False,
                       delete_history: bool = True) -> dict:
        """重置训练状态：清除指定品种/周期的策略和历史，
        使下一次训练从头开始。

        Args:
            symbol: 品种标识。为 None 时清除所有品种。
            timeframe: K 线周期。传入时只清除该周期；为 None 时清除该品种所有周期。
            delete_strategy: 是否删除策略 JSON 文件
            delete_history: 是否删除训练历史 JSON

        Returns:
            {"status": "reset", "deleted": {...}, "message": ...}
        """
        # 训练进行中不允许重置
        with self._lock:
            if self._task and self._task.get("running"):
                return {"status": "error", "message": "训练进行中，请先停止再重置"}

        deleted = {"strategies": [], "histories": []}

        # 构造文件名标记
        parts = []
        if symbol:
            parts.append(symbol)
        if timeframe:
            parts.append(timeframe)
        tag = "_".join(parts) if parts else ""

        # 1. 删除策略 JSON
        if delete_strategy:
            strat_dir = pathlib.Path(Config.STRATEGIES_DIR)
            if strat_dir.exists():
                if tag:
                    target = strat_dir / f"best_{tag}.json"
                    if target.exists():
                        try:
                            target.unlink()
                            deleted["strategies"].append(target.name)
                        except Exception:
                            pass
                else:
                    for f in strat_dir.glob("best_*.json"):
                        try:
                            f.unlink()
                            deleted["strategies"].append(f.name)
                        except Exception:
                            pass

        # 2. 删除训练历史
        if delete_history:
            hist_name = f"training_history_{tag}.json" if tag else "training_history.json"
            # 当不指定 tag 时，删除所有 training_history*.json
            if tag:
                p = pathlib.Path(hist_name)
                if p.exists():
                    try:
                        p.unlink()
                        deleted["histories"].append(p.name)
                    except Exception:
                        pass
            else:
                for p in pathlib.Path(".").glob("training_history*.json"):
                    try:
                        p.unlink()
                        deleted["histories"].append(p.name)
                    except Exception:
                        pass

        # 清空当前任务状态
        with self._lock:
            self._task = None
            self._engine = None

        msg_parts = []
        if deleted["strategies"]:
            msg_parts.append(f"{len(deleted['strategies'])} 个策略")
        if deleted["histories"]:
            msg_parts.append(f"{len(deleted['histories'])} 个历史")
        message = f"已重置训练（{', '.join(msg_parts) if msg_parts else '无文件需删除'}），下次训练将从头开始。"

        return {"status": "reset", "deleted": deleted, "message": message}


# 全局单例
training_service = TrainingService()
