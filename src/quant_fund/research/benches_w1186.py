"""Wave-1186 ux canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.accessibility_studies import bench_accessibility_studies
from quant_fund.models.hci_studies import bench_hci_studies
from quant_fund.models.information_architecture import bench_information_architecture
from quant_fund.models.interaction_design import bench_interaction_design
from quant_fund.models.service_design import bench_service_design
from quant_fund.models.ux_design import bench_ux_design

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


def bench_ux_design_family(seed: int = _SEED + 57400) -> dict[str, float]:
    return _finite_blob(bench_ux_design(seed))


def bench_hci_studies_family(seed: int = _SEED + 57401) -> dict[str, float]:
    return _finite_blob(bench_hci_studies(seed))


def bench_information_architecture_family(seed: int = _SEED + 57402) -> dict[str, float]:
    return _finite_blob(bench_information_architecture(seed))


def bench_interaction_design_family(seed: int = _SEED + 57403) -> dict[str, float]:
    return _finite_blob(bench_interaction_design(seed))


def bench_accessibility_studies_family(seed: int = _SEED + 57404) -> dict[str, float]:
    return _finite_blob(bench_accessibility_studies(seed))


def bench_service_design_family(seed: int = _SEED + 57405) -> dict[str, float]:
    return _finite_blob(bench_service_design(seed))
