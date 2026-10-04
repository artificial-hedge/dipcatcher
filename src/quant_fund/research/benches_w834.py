"""Wave-834 Markov-semigroup bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dirichlet_form import (
    bench_dirichlet_form,
)
from quant_fund.models.hypercontractive import (
    bench_hypercontractive,
)
from quant_fund.models.log_sobolev_sem import (
    bench_log_sobolev_sem,
)
from quant_fund.models.markov_semigroup import (
    bench_markov_semigroup,
)
from quant_fund.models.poincare_semigroup import (
    bench_poincare_semigroup,
)
from quant_fund.models.spectral_gap_sem import (
    bench_spectral_gap_sem,
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


def bench_dirichlet_form_family(
    seed: int = _SEED + 22200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dirichlet_form",
            bench_dirichlet_form(seed),
        )
    )


def bench_markov_semigroup_family(
    seed: int = _SEED + 22201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "markov_semigroup",
            bench_markov_semigroup(seed),
        )
    )


def bench_poincare_semigroup_family(
    seed: int = _SEED + 22202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "poincare_semigroup",
            bench_poincare_semigroup(seed),
        )
    )


def bench_log_sobolev_sem_family(
    seed: int = _SEED + 22203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "log_sobolev_sem",
            bench_log_sobolev_sem(seed),
        )
    )


def bench_hypercontractive_family(
    seed: int = _SEED + 22204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hypercontractive",
            bench_hypercontractive(seed),
        )
    )


def bench_spectral_gap_sem_family(
    seed: int = _SEED + 22205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_gap_sem",
            bench_spectral_gap_sem(seed),
        )
    )
