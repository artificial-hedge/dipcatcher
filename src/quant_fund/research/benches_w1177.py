"""Wave-1177 formal-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.axiomatic_systems import bench_axiomatic_systems
from quant_fund.models.formal_ontology import bench_formal_ontology
from quant_fund.models.formal_sciences import bench_formal_sciences
from quant_fund.models.mathematical_logic import bench_mathematical_logic
from quant_fund.models.model_checking_2 import bench_model_checking_2
from quant_fund.models.proof_calculus import bench_proof_calculus

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


def bench_formal_sciences_family(seed: int = _SEED + 56500) -> dict[str, float]:
    return _finite_blob(bench_formal_sciences(seed))


def bench_mathematical_logic_family(seed: int = _SEED + 56501) -> dict[str, float]:
    return _finite_blob(bench_mathematical_logic(seed))


def bench_axiomatic_systems_family(seed: int = _SEED + 56502) -> dict[str, float]:
    return _finite_blob(bench_axiomatic_systems(seed))


def bench_proof_calculus_family(seed: int = _SEED + 56503) -> dict[str, float]:
    return _finite_blob(bench_proof_calculus(seed))


def bench_model_checking_2_family(seed: int = _SEED + 56504) -> dict[str, float]:
    return _finite_blob(bench_model_checking_2(seed))


def bench_formal_ontology_family(seed: int = _SEED + 56505) -> dict[str, float]:
    return _finite_blob(bench_formal_ontology(seed))
