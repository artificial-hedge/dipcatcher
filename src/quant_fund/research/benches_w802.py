"""Wave-802 neural-SDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.latent_sde import (
    bench_latent_sde,
)
from quant_fund.models.logsig_rde import (
    bench_logsig_rde,
)
from quant_fund.models.neural_cde import (
    bench_neural_cde,
)
from quant_fund.models.neural_rde import (
    bench_neural_rde,
)
from quant_fund.models.sde_gan import (
    bench_sde_gan,
)
from quant_fund.models.sde_matching import (
    bench_sde_matching,
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


def bench_latent_sde_family(
    seed: int = _SEED + 19100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "latent_sde",
            bench_latent_sde(seed),
        )
    )


def bench_neural_cde_family(
    seed: int = _SEED + 19101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "neural_cde",
            bench_neural_cde(seed),
        )
    )


def bench_neural_rde_family(
    seed: int = _SEED + 19102,
) -> dict[str, float]:
    return _floats(_finite_blob("neural_rde", bench_neural_rde(seed)))


def bench_sde_gan_family(
    seed: int = _SEED + 19103,
) -> dict[str, float]:
    return _floats(_finite_blob("sde_gan", bench_sde_gan(seed)))


def bench_sde_matching_family(
    seed: int = _SEED + 19104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sde_matching",
            bench_sde_matching(seed),
        )
    )


def bench_logsig_rde_family(
    seed: int = _SEED + 19105,
) -> dict[str, float]:
    return _floats(_finite_blob("logsig_rde", bench_logsig_rde(seed)))
