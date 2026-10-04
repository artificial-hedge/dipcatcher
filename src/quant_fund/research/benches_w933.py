"""Wave-933 convex-analysis canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bregman_proj import bench_bregman_proj
from quant_fund.models.conjugate_fn import bench_conjugate_fn
from quant_fund.models.fenchel_dual import bench_fenchel_dual
from quant_fund.models.moreau_env import bench_moreau_env
from quant_fund.models.proximal_map import bench_proximal_map
from quant_fund.models.subgradient_proj import bench_subgradient_proj

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


def bench_subgradient_proj_family(seed: int = _SEED + 32100) -> dict[str, float]:
    return _finite_blob(bench_subgradient_proj(seed))


def bench_proximal_map_family(seed: int = _SEED + 32101) -> dict[str, float]:
    return _finite_blob(bench_proximal_map(seed))


def bench_fenchel_dual_family(seed: int = _SEED + 32102) -> dict[str, float]:
    return _finite_blob(bench_fenchel_dual(seed))


def bench_moreau_env_family(seed: int = _SEED + 32103) -> dict[str, float]:
    return _finite_blob(bench_moreau_env(seed))


def bench_bregman_proj_family(seed: int = _SEED + 32104) -> dict[str, float]:
    return _finite_blob(bench_bregman_proj(seed))


def bench_conjugate_fn_family(seed: int = _SEED + 32105) -> dict[str, float]:
    return _finite_blob(bench_conjugate_fn(seed))
