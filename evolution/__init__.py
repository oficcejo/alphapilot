"""
evolution -- Reef Continual Self-Improving Infrastructure (Phase 1: Observe, Phase 2: Adaptive Harness, Phase 3: Grow & Commit)
"""

from .observer import ObserveEngine, get_observer
from .harness import (
    MarketRegime,
    HarnessParams,
    MarketRegimeDetector,
    AdaptiveHarnessPolicy,
    HarnessOptimizer,
    get_harness_policy,
)
from .grow import GrowEngine, get_grow_engine
from .shadow import ShadowEvaluator, get_shadow_evaluator
from .commit import CommitManager, get_commit_manager

__all__ = [
    "ObserveEngine",
    "get_observer",
    "MarketRegime",
    "HarnessParams",
    "MarketRegimeDetector",
    "AdaptiveHarnessPolicy",
    "HarnessOptimizer",
    "get_harness_policy",
    "GrowEngine",
    "get_grow_engine",
    "ShadowEvaluator",
    "get_shadow_evaluator",
    "CommitManager",
    "get_commit_manager",
]
