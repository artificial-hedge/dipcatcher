"""Wave-863 domain-decomposition bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bddc_lite import (
    bench_bddc_lite,
)
from quant_fund.models.coarse_correction import (
    bench_coarse_correction,
)
from quant_fund.models.feti_lite import (
    bench_feti_lite,
)
from quant_fund.models.mortar_dd import (
    bench_mortar_dd,
)
from quant_fund.models.schwarz_add import (
    bench_schwarz_add,
)
from quant_fund.models.schwarz_mult import (
    bench_schwarz_mult,
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


def bench_schwarz_add_family(
    seed: int = _SEED + 25100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schwarz_add",
            bench_schwarz_add(seed),
        )
    )


def bench_schwarz_mult_family(
    seed: int = _SEED + 25101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schwarz_mult",
            bench_schwarz_mult(seed),
        )
    )


def bench_coarse_correction_family(
    seed: int = _SEED + 25102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "coarse_correction",
            bench_coarse_correction(seed),
        )
    )


def bench_mortar_dd_family(
    seed: int = _SEED + 25103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mortar_dd",
            bench_mortar_dd(seed),
        )
    )


def bench_feti_lite_family(
    seed: int = _SEED + 25104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "feti_lite",
            bench_feti_lite(seed),
        )
    )


def bench_bddc_lite_family(
    seed: int = _SEED + 25105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bddc_lite",
            bench_bddc_lite(seed),
        )
    )
