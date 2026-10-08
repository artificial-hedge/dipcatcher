"""Wave-1257 research bench adapter (SYNTHETIC).

clinical-lab canon.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from quant_fund.models.culture_studies import bench_culture_studies
from quant_fund.models.flow_cytometry_studies import bench_flow_cytometry_studies
from quant_fund.models.immunoassay_studies import bench_immunoassay_studies
from quant_fund.models.microscopy_studies import bench_microscopy_studies
from quant_fund.models.pcr_studies import bench_pcr_studies
from quant_fund.models.serology_studies import bench_serology_studies

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in blob.items():
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if not (isinstance(v, float) and math.isfinite(v)):
            raise ValueError(f"metric {k} is not a finite float")
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"metric {k} out of [0,1]")
        out[k] = v
    return out


def _floats(blob: Mapping[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in blob.items()}


def bench_immunoassay_studies_family(seed: int = _SEED + 0) -> dict[str, float]:
    """immunoassay_studies family bench (SYNTHETIC)."""
    return _finite_blob(bench_immunoassay_studies(seed))


def bench_pcr_studies_family(seed: int = _SEED + 100) -> dict[str, float]:
    """pcr_studies family bench (SYNTHETIC)."""
    return _finite_blob(bench_pcr_studies(seed))


def bench_serology_studies_family(seed: int = _SEED + 200) -> dict[str, float]:
    """serology_studies family bench (SYNTHETIC)."""
    return _finite_blob(bench_serology_studies(seed))


def bench_culture_studies_family(seed: int = _SEED + 300) -> dict[str, float]:
    """culture_studies family bench (SYNTHETIC)."""
    return _finite_blob(bench_culture_studies(seed))


def bench_microscopy_studies_family(seed: int = _SEED + 400) -> dict[str, float]:
    """microscopy_studies family bench (SYNTHETIC)."""
    return _finite_blob(bench_microscopy_studies(seed))


def bench_flow_cytometry_studies_family(seed: int = _SEED + 500) -> dict[str, float]:
    """flow_cytometry_studies family bench (SYNTHETIC)."""
    return _finite_blob(bench_flow_cytometry_studies(seed))


__all__ = [
    "bench_immunoassay_studies_family",
    "bench_pcr_studies_family",
    "bench_serology_studies_family",
    "bench_culture_studies_family",
    "bench_microscopy_studies_family",
    "bench_flow_cytometry_studies_family",
]
