"""Wave-554 DT/GW-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.donaldson_thomas import bench_donaldson_thomas
from quant_fund.models.gopakumar_vafa import bench_gopakumar_vafa
from quant_fund.models.gw_descendant import bench_gw_descendant
from quant_fund.models.kontsevich_mgn import bench_kontsevich_mgn
from quant_fund.models.mnop_conj import bench_mnop_conj
from quant_fund.models.pandharipande_thomas import (
    bench_pandharipande_thomas,
)

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


def bench_kontsevich_mgn_family(seed: int = _SEED + 3230) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kontsevich_mgn",
            bench_kontsevich_mgn(seed),
        )
    )


def bench_gw_descendant_family(seed: int = _SEED + 3231) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gw_descendant",
            bench_gw_descendant(seed),
        )
    )


def bench_donaldson_thomas_family(
    seed: int = _SEED + 3232,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "donaldson_thomas",
            bench_donaldson_thomas(seed),
        )
    )


def bench_pandharipande_thomas_family(
    seed: int = _SEED + 3233,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pandharipande_thomas",
            bench_pandharipande_thomas(seed),
        )
    )


def bench_gopakumar_vafa_family(
    seed: int = _SEED + 3234,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gopakumar_vafa",
            bench_gopakumar_vafa(seed),
        )
    )


def bench_mnop_conj_family(seed: int = _SEED + 3235) -> dict[str, float]:
    return _floats(_finite_blob("mnop_conj", bench_mnop_conj(seed)))
