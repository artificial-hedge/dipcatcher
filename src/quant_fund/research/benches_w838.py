"""Wave-838 discrete-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.caratheodory_thm import (
    bench_caratheodory_thm,
)
from quant_fund.models.farkas_lemma import (
    bench_farkas_lemma,
)
from quant_fund.models.lattice_point import (
    bench_lattice_point,
)
from quant_fund.models.radon_theorem import (
    bench_radon_theorem,
)
from quant_fund.models.separation_thm import (
    bench_separation_thm,
)
from quant_fund.models.tverberg_thm import (
    bench_tverberg_thm,
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


def bench_radon_theorem_family(
    seed: int = _SEED + 22600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "radon_theorem",
            bench_radon_theorem(seed),
        )
    )


def bench_caratheodory_thm_family(
    seed: int = _SEED + 22601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "caratheodory_thm",
            bench_caratheodory_thm(seed),
        )
    )


def bench_farkas_lemma_family(
    seed: int = _SEED + 22602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "farkas_lemma",
            bench_farkas_lemma(seed),
        )
    )


def bench_separation_thm_family(
    seed: int = _SEED + 22603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "separation_thm",
            bench_separation_thm(seed),
        )
    )


def bench_lattice_point_family(
    seed: int = _SEED + 22604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lattice_point",
            bench_lattice_point(seed),
        )
    )


def bench_tverberg_thm_family(
    seed: int = _SEED + 22605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tverberg_thm",
            bench_tverberg_thm(seed),
        )
    )
