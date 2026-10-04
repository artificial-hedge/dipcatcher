"""Wave-1010 quantum-field-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.canonical_quantization import bench_canonical_quantization
from quant_fund.models.dirac_equation import bench_dirac_equation
from quant_fund.models.feynman_rules import bench_feynman_rules
from quant_fund.models.klein_gordon import bench_klein_gordon
from quant_fund.models.path_integral_qm import bench_path_integral_qm
from quant_fund.models.renormalization_group import bench_renormalization_group

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


def bench_klein_gordon_family(seed: int = _SEED + 39800) -> dict[str, float]:
    return _finite_blob(bench_klein_gordon(seed))


def bench_dirac_equation_family(seed: int = _SEED + 39801) -> dict[str, float]:
    return _finite_blob(bench_dirac_equation(seed))


def bench_feynman_rules_family(seed: int = _SEED + 39802) -> dict[str, float]:
    return _finite_blob(bench_feynman_rules(seed))


def bench_renormalization_group_family(seed: int = _SEED + 39803) -> dict[str, float]:
    return _finite_blob(bench_renormalization_group(seed))


def bench_path_integral_qm_family(seed: int = _SEED + 39804) -> dict[str, float]:
    return _finite_blob(bench_path_integral_qm(seed))


def bench_canonical_quantization_family(seed: int = _SEED + 39805) -> dict[str, float]:
    return _finite_blob(bench_canonical_quantization(seed))
