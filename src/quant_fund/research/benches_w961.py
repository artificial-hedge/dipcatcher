"""Wave-961 semigroup-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.analytic_semigroup import bench_analytic_semigroup
from quant_fund.models.c0_semigroup import bench_c0_semigroup
from quant_fund.models.cosine_family import bench_cosine_family
from quant_fund.models.hille_yosida import bench_hille_yosida
from quant_fund.models.lumer_phillips import bench_lumer_phillips
from quant_fund.models.trotter_kato import bench_trotter_kato

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


def bench_c0_semigroup_family(seed: int = _SEED + 34900) -> dict[str, float]:
    return _finite_blob(bench_c0_semigroup(seed))


def bench_hille_yosida_family(seed: int = _SEED + 34901) -> dict[str, float]:
    return _finite_blob(bench_hille_yosida(seed))


def bench_lumer_phillips_family(seed: int = _SEED + 34902) -> dict[str, float]:
    return _finite_blob(bench_lumer_phillips(seed))


def bench_analytic_semigroup_family(seed: int = _SEED + 34903) -> dict[str, float]:
    return _finite_blob(bench_analytic_semigroup(seed))


def bench_cosine_family_family(seed: int = _SEED + 34904) -> dict[str, float]:
    return _finite_blob(bench_cosine_family(seed))


def bench_trotter_kato_family(seed: int = _SEED + 34905) -> dict[str, float]:
    return _finite_blob(bench_trotter_kato(seed))
