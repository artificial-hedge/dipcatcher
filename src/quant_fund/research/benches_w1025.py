"""Wave-1025 social-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.behavioral_econ import bench_behavioral_econ
from quant_fund.models.cognitive_science import bench_cognitive_science
from quant_fund.models.game_theory2 import bench_game_theory2
from quant_fund.models.linguistics import bench_linguistics
from quant_fund.models.political_science import bench_political_science
from quant_fund.models.sociology_net import bench_sociology_net

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


def bench_game_theory2_family(seed: int = _SEED + 41300) -> dict[str, float]:
    return _finite_blob(bench_game_theory2(seed))


def bench_behavioral_econ_family(seed: int = _SEED + 41301) -> dict[str, float]:
    return _finite_blob(bench_behavioral_econ(seed))


def bench_political_science_family(seed: int = _SEED + 41302) -> dict[str, float]:
    return _finite_blob(bench_political_science(seed))


def bench_sociology_net_family(seed: int = _SEED + 41303) -> dict[str, float]:
    return _finite_blob(bench_sociology_net(seed))


def bench_cognitive_science_family(seed: int = _SEED + 41304) -> dict[str, float]:
    return _finite_blob(bench_cognitive_science(seed))


def bench_linguistics_family(seed: int = _SEED + 41305) -> dict[str, float]:
    return _finite_blob(bench_linguistics(seed))
