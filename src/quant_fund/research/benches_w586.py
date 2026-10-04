"""Wave-586 duality-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dualizing_cmplx import (
    bench_dualizing_cmplx,
)
from quant_fund.models.dualizing_sheaf import (
    bench_dualizing_sheaf,
)
from quant_fund.models.groth_duality import (
    bench_groth_duality,
)
from quant_fund.models.relative_duality import (
    bench_relative_duality,
)
from quant_fund.models.residue_thm import bench_residue_thm
from quant_fund.models.verdier_duality import (
    bench_verdier_duality,
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


def bench_groth_duality_family(
    seed: int = _SEED + 3422,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "groth_duality",
            bench_groth_duality(seed),
        )
    )


def bench_dualizing_cmplx_family(
    seed: int = _SEED + 3423,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dualizing_cmplx",
            bench_dualizing_cmplx(seed),
        )
    )


def bench_residue_thm_family(seed: int = _SEED + 3424) -> dict[str, float]:
    return _floats(_finite_blob("residue_thm", bench_residue_thm(seed)))


def bench_verdier_duality_family(
    seed: int = _SEED + 3425,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "verdier_duality",
            bench_verdier_duality(seed),
        )
    )


def bench_dualizing_sheaf_family(
    seed: int = _SEED + 3426,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dualizing_sheaf",
            bench_dualizing_sheaf(seed),
        )
    )


def bench_relative_duality_family(
    seed: int = _SEED + 3427,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "relative_duality",
            bench_relative_duality(seed),
        )
    )
