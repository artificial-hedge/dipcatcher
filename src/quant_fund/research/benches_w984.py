"""Wave-984 modulation-spaces canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ambiguity_fn import bench_ambiguity_fn
from quant_fund.models.feichtinger_alg import bench_feichtinger_alg
from quant_fund.models.gabor_frame import bench_gabor_frame
from quant_fund.models.modulation_space import bench_modulation_space
from quant_fund.models.short_time_ft import bench_short_time_ft
from quant_fund.models.wigner_dist import bench_wigner_dist

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


def bench_modulation_space_family(seed: int = _SEED + 37200) -> dict[str, float]:
    return _finite_blob(bench_modulation_space(seed))


def bench_short_time_ft_family(seed: int = _SEED + 37201) -> dict[str, float]:
    return _finite_blob(bench_short_time_ft(seed))


def bench_gabor_frame_family(seed: int = _SEED + 37202) -> dict[str, float]:
    return _finite_blob(bench_gabor_frame(seed))


def bench_wigner_dist_family(seed: int = _SEED + 37203) -> dict[str, float]:
    return _finite_blob(bench_wigner_dist(seed))


def bench_ambiguity_fn_family(seed: int = _SEED + 37204) -> dict[str, float]:
    return _finite_blob(bench_ambiguity_fn(seed))


def bench_feichtinger_alg_family(seed: int = _SEED + 37205) -> dict[str, float]:
    return _finite_blob(bench_feichtinger_alg(seed))
