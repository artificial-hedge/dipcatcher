"""Wave-595 crystalline bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.conjugate_fil import (
    bench_conjugate_fil,
)
from quant_fund.models.crys_cohom import bench_crys_cohom
from quant_fund.models.divided_power import (
    bench_divided_power,
)
from quant_fund.models.nygaard_filt import (
    bench_nygaard_filt,
)
from quant_fund.models.pd_envelope import bench_pd_envelope
from quant_fund.models.syntomic import bench_syntomic

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


def bench_crys_cohom_family(
    seed: int = _SEED + 3476,
) -> dict[str, float]:
    return _floats(_finite_blob("crys_cohom", bench_crys_cohom(seed)))


def bench_syntomic_family(
    seed: int = _SEED + 3477,
) -> dict[str, float]:
    return _floats(_finite_blob("syntomic", bench_syntomic(seed)))


def bench_divided_power_family(
    seed: int = _SEED + 3478,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "divided_power",
            bench_divided_power(seed),
        )
    )


def bench_pd_envelope_family(
    seed: int = _SEED + 3479,
) -> dict[str, float]:
    return _floats(_finite_blob("pd_envelope", bench_pd_envelope(seed)))


def bench_nygaard_filt_family(
    seed: int = _SEED + 3480,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nygaard_filt",
            bench_nygaard_filt(seed),
        )
    )


def bench_conjugate_fil_family(
    seed: int = _SEED + 3481,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "conjugate_fil",
            bench_conjugate_fil(seed),
        )
    )
