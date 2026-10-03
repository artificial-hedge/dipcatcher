"""Wave-724 arithmetic-cycles bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coates_wiles import bench_coates_wiles
from quant_fund.models.gan_gross_prasad import (
    bench_gan_gross_prasad,
)
from quant_fund.models.greenberg_selmer import (
    bench_greenberg_selmer,
)
from quant_fund.models.heegner_cycle import bench_heegner_cycle
from quant_fund.models.iwasawa_lfunc import bench_iwasawa_lfunc
from quant_fund.models.kurihara_iwasawa import (
    bench_kurihara_iwasawa,
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


def bench_coates_wiles_family(
    seed: int = _SEED + 11300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "coates_wiles",
            bench_coates_wiles(seed),
        )
    )


def bench_iwasawa_lfunc_family(
    seed: int = _SEED + 11301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "iwasawa_lfunc",
            bench_iwasawa_lfunc(seed),
        )
    )


def bench_greenberg_selmer_family(
    seed: int = _SEED + 11302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "greenberg_selmer",
            bench_greenberg_selmer(seed),
        )
    )


def bench_kurihara_iwasawa_family(
    seed: int = _SEED + 11303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kurihara_iwasawa",
            bench_kurihara_iwasawa(seed),
        )
    )


def bench_heegner_cycle_family(
    seed: int = _SEED + 11304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "heegner_cycle",
            bench_heegner_cycle(seed),
        )
    )


def bench_gan_gross_prasad_family(
    seed: int = _SEED + 11305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gan_gross_prasad",
            bench_gan_gross_prasad(seed),
        )
    )
