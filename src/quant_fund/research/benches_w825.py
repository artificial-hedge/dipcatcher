"""Wave-825 measurable-selection bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.castaing_rep import (
    bench_castaing_rep,
)
from quant_fund.models.integrand_map import (
    bench_integrand_map,
)
from quant_fund.models.kura_ryll import (
    bench_kura_ryll,
)
from quant_fund.models.measur_select import (
    bench_measur_select,
)
from quant_fund.models.measurable_graph import (
    bench_measurable_graph,
)
from quant_fund.models.stoch_open import (
    bench_stoch_open,
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


def bench_measur_select_family(
    seed: int = _SEED + 21300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "measur_select",
            bench_measur_select(seed),
        )
    )


def bench_kura_ryll_family(
    seed: int = _SEED + 21301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kura_ryll",
            bench_kura_ryll(seed),
        )
    )


def bench_castaing_rep_family(
    seed: int = _SEED + 21302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "castaing_rep",
            bench_castaing_rep(seed),
        )
    )


def bench_measurable_graph_family(
    seed: int = _SEED + 21303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "measurable_graph",
            bench_measurable_graph(seed),
        )
    )


def bench_integrand_map_family(
    seed: int = _SEED + 21304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "integrand_map",
            bench_integrand_map(seed),
        )
    )


def bench_stoch_open_family(
    seed: int = _SEED + 21305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stoch_open",
            bench_stoch_open(seed),
        )
    )
