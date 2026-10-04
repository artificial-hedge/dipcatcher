"""Wave-721 mixed-motives bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beilinson_height import (
    bench_beilinson_height,
)
from quant_fund.models.brown_motives import bench_brown_motives
from quant_fund.models.mixed_elliptic import (
    bench_mixed_elliptic,
)
from quant_fund.models.motivic_pi import bench_motivic_pi
from quant_fund.models.mzc_motive import bench_mzc_motive
from quant_fund.models.zeta_element import bench_zeta_element

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


def bench_brown_motives_family(
    seed: int = _SEED + 11000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "brown_motives",
            bench_brown_motives(seed),
        )
    )


def bench_mzc_motive_family(
    seed: int = _SEED + 11001,
) -> dict[str, float]:
    return _floats(_finite_blob("mzc_motive", bench_mzc_motive(seed)))


def bench_zeta_element_family(
    seed: int = _SEED + 11002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zeta_element",
            bench_zeta_element(seed),
        )
    )


def bench_mixed_elliptic_family(
    seed: int = _SEED + 11003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mixed_elliptic",
            bench_mixed_elliptic(seed),
        )
    )


def bench_motivic_pi_family(
    seed: int = _SEED + 11004,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_pi", bench_motivic_pi(seed)))


def bench_beilinson_height_family(
    seed: int = _SEED + 11005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "beilinson_height",
            bench_beilinson_height(seed),
        )
    )
