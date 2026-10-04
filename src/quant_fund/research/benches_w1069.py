"""Wave-1069 criminal justice canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.criminal_justice import bench_criminal_justice
from quant_fund.models.criminal_procedure import bench_criminal_procedure
from quant_fund.models.forensic_science import bench_forensic_science
from quant_fund.models.penology import bench_penology
from quant_fund.models.policing_studies import bench_policing_studies
from quant_fund.models.victimology import bench_victimology

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


def bench_criminal_justice_family(seed: int = _SEED + 45700) -> dict[str, float]:
    return _finite_blob(bench_criminal_justice(seed))


def bench_forensic_science_family(seed: int = _SEED + 45701) -> dict[str, float]:
    return _finite_blob(bench_forensic_science(seed))


def bench_penology_family(seed: int = _SEED + 45702) -> dict[str, float]:
    return _finite_blob(bench_penology(seed))


def bench_policing_studies_family(seed: int = _SEED + 45703) -> dict[str, float]:
    return _finite_blob(bench_policing_studies(seed))


def bench_victimology_family(seed: int = _SEED + 45704) -> dict[str, float]:
    return _finite_blob(bench_victimology(seed))


def bench_criminal_procedure_family(seed: int = _SEED + 45705) -> dict[str, float]:
    return _finite_blob(bench_criminal_procedure(seed))
