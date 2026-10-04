"""Wave-1053 psychology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.behavioral_neuroscience import bench_behavioral_neuroscience
from quant_fund.models.clinical_psychology import bench_clinical_psychology
from quant_fund.models.cognitive_psychology import bench_cognitive_psychology
from quant_fund.models.developmental_psychology import bench_developmental_psychology
from quant_fund.models.psychometrics import bench_psychometrics
from quant_fund.models.social_psychology import bench_social_psychology

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


def bench_cognitive_psychology_family(seed: int = _SEED + 44100) -> dict[str, float]:
    return _finite_blob(bench_cognitive_psychology(seed))


def bench_psychometrics_family(seed: int = _SEED + 44101) -> dict[str, float]:
    return _finite_blob(bench_psychometrics(seed))


def bench_behavioral_neuroscience_family(seed: int = _SEED + 44102) -> dict[str, float]:
    return _finite_blob(bench_behavioral_neuroscience(seed))


def bench_social_psychology_family(seed: int = _SEED + 44103) -> dict[str, float]:
    return _finite_blob(bench_social_psychology(seed))


def bench_developmental_psychology_family(seed: int = _SEED + 44104) -> dict[str, float]:
    return _finite_blob(bench_developmental_psychology(seed))


def bench_clinical_psychology_family(seed: int = _SEED + 44105) -> dict[str, float]:
    return _finite_blob(bench_clinical_psychology(seed))
