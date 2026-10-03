"""Wave-33 optional benchmark adapters.

Each adapter runs one wave-33 module's self-check and returns a
``{metric_key: float}`` blob — all ``synthetic_*`` proper diagnostics, never
market evidence. A lane that is unavailable (ImportError) or whose numbers are
non-finite — the ``_finite_blob`` gate, waves 15-32 precedent — degrades to
``{}`` so the scorecard omits it cleanly.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-33 stamp seed

# --- per-family seeds -------------------------------------------------------
_CAUSAL_IMPACT_SEED = _SEED + 185
_WEAK_IV_SEED = _SEED + 186
_SYNTH_DID_SEED = _SEED + 187
_PERMUTATION_INFERENCE_SEED = _SEED + 188
_PROPENSITY_SCORE_SEED = _SEED + 189
_CLUSTER_ROBUST_SEED = _SEED + 190


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(
    raw: dict[str, float] | dict[str, object],
) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_causal_impact() -> dict[str, float]:
    try:
        from quant_fund.models.causal_impact import bench_causal_impact as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CAUSAL_IMPACT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_weak_iv() -> dict[str, float]:
    try:
        from quant_fund.models.weak_iv import bench_weak_iv as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_WEAK_IV_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_synth_did() -> dict[str, float]:
    try:
        from quant_fund.models.synth_did import bench_synth_did as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SYNTH_DID_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_permutation_inference() -> dict[str, float]:
    try:
        from quant_fund.models.permutation_inference import (
            bench_permutation_inference as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PERMUTATION_INFERENCE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_propensity_score() -> dict[str, float]:
    try:
        from quant_fund.models.propensity_score import (
            bench_propensity_score as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PROPENSITY_SCORE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_cluster_robust() -> dict[str, float]:
    try:
        from quant_fund.models.cluster_robust import bench_cluster_robust as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CLUSTER_ROBUST_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
