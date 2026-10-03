"""Wave-983 Besov/Triebel-Lizorkin canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atoms_decomp import bench_atoms_decomp
from quant_fund.models.besov_embed import bench_besov_embed
from quant_fund.models.besov_space import bench_besov_space
from quant_fund.models.hardy_littlewood_max import bench_hardy_littlewood_max
from quant_fund.models.triebel_lizorkin import bench_triebel_lizorkin
from quant_fund.models.wavelet_char import bench_wavelet_char

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


def bench_besov_space_family(seed: int = _SEED + 37100) -> dict[str, float]:
    return _finite_blob(bench_besov_space(seed))


def bench_triebel_lizorkin_family(seed: int = _SEED + 37101) -> dict[str, float]:
    return _finite_blob(bench_triebel_lizorkin(seed))


def bench_atoms_decomp_family(seed: int = _SEED + 37102) -> dict[str, float]:
    return _finite_blob(bench_atoms_decomp(seed))


def bench_wavelet_char_family(seed: int = _SEED + 37103) -> dict[str, float]:
    return _finite_blob(bench_wavelet_char(seed))


def bench_besov_embed_family(seed: int = _SEED + 37104) -> dict[str, float]:
    return _finite_blob(bench_besov_embed(seed))


def bench_hardy_littlewood_max_family(seed: int = _SEED + 37105) -> dict[str, float]:
    return _finite_blob(bench_hardy_littlewood_max(seed))
