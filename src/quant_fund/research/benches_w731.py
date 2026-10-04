"""Wave-731 ramification-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brylinski_kato import (
    bench_brylinski_kato,
)
from quant_fund.models.higher_ramif import bench_higher_ramif
from quant_fund.models.log_ramification import (
    bench_log_ramification,
)
from quant_fund.models.neron_raynaud import (
    bench_neron_raynaud,
)
from quant_fund.models.semi_stable_model import (
    bench_semi_stable_model,
)
from quant_fund.models.temkin_alter import (
    bench_temkin_alter,
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


def bench_higher_ramif_family(
    seed: int = _SEED + 12000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higher_ramif",
            bench_higher_ramif(seed),
        )
    )


def bench_brylinski_kato_family(
    seed: int = _SEED + 12001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "brylinski_kato",
            bench_brylinski_kato(seed),
        )
    )


def bench_log_ramification_family(
    seed: int = _SEED + 12002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "log_ramification",
            bench_log_ramification(seed),
        )
    )


def bench_semi_stable_model_family(
    seed: int = _SEED + 12003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "semi_stable_model",
            bench_semi_stable_model(seed),
        )
    )


def bench_neron_raynaud_family(
    seed: int = _SEED + 12004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "neron_raynaud",
            bench_neron_raynaud(seed),
        )
    )


def bench_temkin_alter_family(
    seed: int = _SEED + 12005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "temkin_alter",
            bench_temkin_alter(seed),
        )
    )
