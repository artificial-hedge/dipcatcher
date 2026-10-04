"""Wave-925 Bayesian-nonparametrics-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bondesson_shot import bench_bondesson_shot
from quant_fund.models.exchangeable_pf import bench_exchangeable_pf
from quant_fund.models.kingman_paintbox import bench_kingman_paintbox
from quant_fund.models.nggp_process import bench_nggp_process
from quant_fund.models.normalized_rm import bench_normalized_rm
from quant_fund.models.sigma_stable import bench_sigma_stable

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_exchangeable_pf_family(seed: int = _SEED + 31300) -> dict[str, float]:
    return _finite_blob(bench_exchangeable_pf(seed))


def bench_normalized_rm_family(seed: int = _SEED + 31301) -> dict[str, float]:
    return _finite_blob(bench_normalized_rm(seed))


def bench_sigma_stable_family(seed: int = _SEED + 31302) -> dict[str, float]:
    return _finite_blob(bench_sigma_stable(seed))


def bench_nggp_process_family(seed: int = _SEED + 31303) -> dict[str, float]:
    return _finite_blob(bench_nggp_process(seed))


def bench_bondesson_shot_family(seed: int = _SEED + 31304) -> dict[str, float]:
    return _finite_blob(bench_bondesson_shot(seed))


def bench_kingman_paintbox_family(seed: int = _SEED + 31305) -> dict[str, float]:
    return _finite_blob(bench_kingman_paintbox(seed))
