"""Wave-660 chromatic-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chromatic_completion import bench_chromatic_completion
from quant_fund.models.chromatic_l2 import bench_chromatic_l2
from quant_fund.models.morava_k2 import bench_morava_k2
from quant_fund.models.periodicity_height import bench_periodicity_height
from quant_fund.models.picard_spec import bench_picard_spec
from quant_fund.models.telescope_tower2 import bench_telescope_tower2

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_morava_k2_family(seed: int = _SEED + 4900) -> dict[str, float]:
    return _floats(_finite_blob("morava_k2", bench_morava_k2(seed)))


def bench_telescope_tower2_family(
    seed: int = _SEED + 4901,
) -> dict[str, float]:
    return _floats(_finite_blob("telescope_tower2", bench_telescope_tower2(seed)))


def bench_chromatic_l2_family(
    seed: int = _SEED + 4902,
) -> dict[str, float]:
    return _floats(_finite_blob("chromatic_l2", bench_chromatic_l2(seed)))


def bench_picard_spec_family(
    seed: int = _SEED + 4903,
) -> dict[str, float]:
    return _floats(_finite_blob("picard_spec", bench_picard_spec(seed)))


def bench_periodicity_height_family(
    seed: int = _SEED + 4904,
) -> dict[str, float]:
    return _floats(_finite_blob("periodicity_height", bench_periodicity_height(seed)))


def bench_chromatic_completion_family(
    seed: int = _SEED + 4905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_completion",
            bench_chromatic_completion(seed),
        )
    )
