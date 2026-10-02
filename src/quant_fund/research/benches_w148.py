"""Wave-126 adapters: exec-summary interpretability canon — sae_feature,
activation_steering, probe_linear, logit_lens, patch_activation, circuit_ablation —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.activation_steering import bench_activation_steering
from quant_fund.models.circuit_ablation import bench_circuit_ablation
from quant_fund.models.logit_lens import bench_logit_lens
from quant_fund.models.patch_activation import bench_patch_activation
from quant_fund.models.probe_linear import bench_probe_linear
from quant_fund.models.sae_feature import bench_sae_feature

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_sae_feature_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sae_feature", bench_sae_feature(seed=_SEED + 876)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sae_feature bench failed: {exc}") from exc


def bench_activation_steering_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("activation_steering", bench_activation_steering(seed=_SEED + 877))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"activation_steering bench failed: {exc}") from exc


def bench_probe_linear_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("probe_linear", bench_probe_linear(seed=_SEED + 878)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"probe_linear bench failed: {exc}") from exc


def bench_logit_lens_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("logit_lens", bench_logit_lens(seed=_SEED + 879)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"logit_lens bench failed: {exc}") from exc


def bench_patch_activation_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("patch_activation", bench_patch_activation(seed=_SEED + 880)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"patch_activation bench failed: {exc}") from exc


def bench_circuit_ablation_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("circuit_ablation", bench_circuit_ablation(seed=_SEED + 881)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"circuit_ablation bench failed: {exc}") from exc
