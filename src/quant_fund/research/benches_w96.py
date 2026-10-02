"""Wave-96 adapters: classic ML canon IV — Gaussian/
multinomial/Bernoulli naive Bayes, AdaBoost.M1 + LogitBoost
stumps, expectation propagation for Bayesian probit, item-kNN
+ ALS-WR + bias-MF collaborative filtering, Apriori
association rules, and fictitious-play / support-enumeration
/ regret-matching game solvers.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adaboost import bench_adaboost
from quant_fund.models.association_rules import bench_association_rules
from quant_fund.models.collaborative_filtering import (
    bench_collaborative_filtering,
)
from quant_fund.models.expectation_propagation import (
    bench_expectation_propagation,
)
from quant_fund.models.naive_bayes import bench_naive_bayes
from quant_fund.models.nash_equilibrium import bench_nash_equilibrium

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"non-finite {k} in {name}")
        flat[k] = float(arr.ravel()[0])
    if not flat:
        raise ValueError(f"{name} returned no metrics")
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_naive_bayes_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("naive_bayes", bench_naive_bayes(seed=_SEED + 564)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"naive_bayes bench failed: {exc}") from exc


def bench_adaboost_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("adaboost", bench_adaboost(seed=_SEED + 565)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adaboost bench failed: {exc}") from exc


def bench_expectation_propagation_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob(
                "expectation_propagation",
                bench_expectation_propagation(seed=_SEED + 566),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"expectation_propagation bench failed: {exc}") from exc


def bench_collaborative_filtering_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob(
                "collaborative_filtering",
                bench_collaborative_filtering(seed=_SEED + 567),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"collaborative_filtering bench failed: {exc}") from exc


def bench_association_rules_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("association_rules", bench_association_rules(seed=_SEED + 568))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"association_rules bench failed: {exc}") from exc


def bench_nash_equilibrium_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("nash_equilibrium", bench_nash_equilibrium(seed=_SEED + 569))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nash_equilibrium bench failed: {exc}") from exc
