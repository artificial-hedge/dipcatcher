"""Wave-92 bench adapters: metaheuristic optimizers,
Bayesian optimization, fuzzy clustering, self-organizing
maps + LVQ, PageRank link topology, and Hyperband
multi-fidelity search — SYNTHETIC self-checks emitting
proper scores only."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bayesian_optimization import bench_bayes_opt
from quant_fund.models.fuzzy_clustering import bench_fuzzy
from quant_fund.models.hyperband_search import bench_hyperband
from quant_fund.models.metaheuristic_optimizers import bench_metaheuristics
from quant_fund.models.pagerank_topology import bench_pagerank
from quant_fund.models.self_organizing_maps import bench_som

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}

_META_SEED = 20261231 + 540
_BAYESOPT_SEED = 20261231 + 541
_FUZZY_SEED = 20261231 + 542
_SOM_SEED = 20261231 + 543
_PAGERANK_SEED = 20261231 + 544
_HYPERBAND_SEED = 20261231 + 545


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            flat[f"{name}_{k}"] = float(v)
    if not flat:
        raise ValueError(f"{name}: no finite scalar emissions")
    if any(tok in k for k in flat for tok in _FORBIDDEN):
        raise ValueError(f"{name}: forbidden metric token emitted")
    return flat


def _isinstance_floats(flat: dict[str, float]) -> dict[str, float]:
    if not all(isinstance(v, float) for v in flat.values()):
        raise ValueError("non-float emission")
    return flat


def bench_metaheuristic_optimizers() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("metaheur", bench_metaheuristics(seed=_META_SEED)))
    except ImportError as e:
        raise ImportError(f"metaheur bench unavailable: {e}") from e
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) as e:
        raise ValueError(f"metaheur bench failed: {e}") from e


def bench_bayesian_optimization() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("bayesopt", bench_bayes_opt(seed=_BAYESOPT_SEED)))
    except ImportError as e:
        raise ImportError(f"bayesopt bench unavailable: {e}") from e
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) as e:
        raise ValueError(f"bayesopt bench failed: {e}") from e


def bench_fuzzy_clustering() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("fuzzy", bench_fuzzy(seed=_FUZZY_SEED)))
    except ImportError as e:
        raise ImportError(f"fuzzy bench unavailable: {e}") from e
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) as e:
        raise ValueError(f"fuzzy bench failed: {e}") from e


def bench_self_organizing_maps() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("som", bench_som(seed=_SOM_SEED)))
    except ImportError as e:
        raise ImportError(f"som bench unavailable: {e}") from e
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) as e:
        raise ValueError(f"som bench failed: {e}") from e


def bench_pagerank_topology() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("pagerank", bench_pagerank(seed=_PAGERANK_SEED)))
    except ImportError as e:
        raise ImportError(f"pagerank bench unavailable: {e}") from e
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) as e:
        raise ValueError(f"pagerank bench failed: {e}") from e


def bench_hyperband_search() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("hyperband", bench_hyperband(seed=_HYPERBAND_SEED)))
    except ImportError as e:
        raise ImportError(f"hyperband bench unavailable: {e}") from e
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) as e:
        raise ValueError(f"hyperband bench failed: {e}") from e
