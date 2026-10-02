"""Wave-126 adapters: exec-summary energy-based-model canon — persistent_cd,
score_matching, contrastive_divergence, denoising_sm, noise_contrastive, adversarial_ebm —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adversarial_ebm import bench_adversarial_ebm
from quant_fund.models.contrastive_divergence import bench_contrastive_divergence
from quant_fund.models.denoising_sm import bench_denoising_sm
from quant_fund.models.noise_contrastive import bench_noise_contrastive
from quant_fund.models.persistent_cd import bench_persistent_cd
from quant_fund.models.score_matching import bench_score_matching

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


def bench_persistent_cd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("persistent_cd", bench_persistent_cd(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"persistent_cd bench failed: {exc}") from exc


def bench_score_matching_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("score_matching", bench_score_matching(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"score_matching bench failed: {exc}") from exc


def bench_contrastive_divergence_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("contrastive_divergence", bench_contrastive_divergence(seed=_SEED + 962))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"contrastive_divergence bench failed: {exc}") from exc


def bench_denoising_sm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("denoising_sm", bench_denoising_sm(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"denoising_sm bench failed: {exc}") from exc


def bench_noise_contrastive_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("noise_contrastive", bench_noise_contrastive(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"noise_contrastive bench failed: {exc}") from exc


def bench_adversarial_ebm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("adversarial_ebm", bench_adversarial_ebm(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adversarial_ebm bench failed: {exc}") from exc
