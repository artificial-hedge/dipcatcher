"""Wave-540 transcendence-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.baker_thm import bench_baker_thm
from quant_fund.models.gelfond_schneider import bench_gelfond_schneider
from quant_fund.models.hermite_lindemann import bench_hermite_lindemann
from quant_fund.models.lindemann_weier import bench_lindemann_weier
from quant_fund.models.schanuel_conj import bench_schanuel_conj
from quant_fund.models.siegel_shidlovskii import bench_siegel_shidlovskii

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


def bench_hermite_lindemann_family(
    seed: int = _SEED + 3146,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hermite_lindemann",
            bench_hermite_lindemann(seed),
        )
    )


def bench_gelfond_schneider_family(
    seed: int = _SEED + 3147,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gelfond_schneider",
            bench_gelfond_schneider(seed),
        )
    )


def bench_baker_thm_family(seed: int = _SEED + 3148) -> dict[str, float]:
    return _floats(_finite_blob("baker_thm", bench_baker_thm(seed)))


def bench_lindemann_weier_family(
    seed: int = _SEED + 3149,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lindemann_weier",
            bench_lindemann_weier(seed),
        )
    )


def bench_schanuel_conj_family(seed: int = _SEED + 3150) -> dict[str, float]:
    return _floats(_finite_blob("schanuel_conj", bench_schanuel_conj(seed)))


def bench_siegel_shidlovskii_family(
    seed: int = _SEED + 3151,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "siegel_shidlovskii",
            bench_siegel_shidlovskii(seed),
        )
    )
