"""Wave-558 gauge-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anti_self_dual import bench_anti_self_dual
from quant_fund.models.higgs_bundle import bench_higgs_bundle
from quant_fund.models.instanton_moduli import bench_instanton_moduli
from quant_fund.models.kapustin_witten import bench_kapustin_witten
from quant_fund.models.nahm_transform import bench_nahm_transform
from quant_fund.models.yang_mills import bench_yang_mills

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


def bench_yang_mills_family(seed: int = _SEED + 3254) -> dict[str, float]:
    return _floats(_finite_blob("yang_mills", bench_yang_mills(seed)))


def bench_instanton_moduli_family(
    seed: int = _SEED + 3255,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "instanton_moduli",
            bench_instanton_moduli(seed),
        )
    )


def bench_anti_self_dual_family(seed: int = _SEED + 3256) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "anti_self_dual",
            bench_anti_self_dual(seed),
        )
    )


def bench_higgs_bundle_family(seed: int = _SEED + 3257) -> dict[str, float]:
    return _floats(_finite_blob("higgs_bundle", bench_higgs_bundle(seed)))


def bench_kapustin_witten_family(
    seed: int = _SEED + 3258,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kapustin_witten",
            bench_kapustin_witten(seed),
        )
    )


def bench_nahm_transform_family(seed: int = _SEED + 3259) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nahm_transform",
            bench_nahm_transform(seed),
        )
    )
