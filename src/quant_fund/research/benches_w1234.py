"""Wave-1234 womens-health canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.breast_medicine import bench_breast_medicine
from quant_fund.models.contraception_studies import bench_contraception_studies
from quant_fund.models.infertility_studies import bench_infertility_studies
from quant_fund.models.menopause_medicine import bench_menopause_medicine
from quant_fund.models.pelvic_health_studies import bench_pelvic_health_studies
from quant_fund.models.urogynecology_studies import bench_urogynecology_studies

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


def bench_menopause_medicine_family(seed: int = _SEED + 62200) -> dict[str, float]:
    return _finite_blob(bench_menopause_medicine(seed))


def bench_urogynecology_studies_family(seed: int = _SEED + 62201) -> dict[str, float]:
    return _finite_blob(bench_urogynecology_studies(seed))


def bench_breast_medicine_family(seed: int = _SEED + 62202) -> dict[str, float]:
    return _finite_blob(bench_breast_medicine(seed))


def bench_infertility_studies_family(seed: int = _SEED + 62203) -> dict[str, float]:
    return _finite_blob(bench_infertility_studies(seed))


def bench_contraception_studies_family(seed: int = _SEED + 62204) -> dict[str, float]:
    return _finite_blob(bench_contraception_studies(seed))


def bench_pelvic_health_studies_family(seed: int = _SEED + 62205) -> dict[str, float]:
    return _finite_blob(bench_pelvic_health_studies(seed))
