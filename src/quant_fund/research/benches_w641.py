"""Wave-641 homotopy-17 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chromatic_htpy import (
    bench_chromatic_htpy,
)
from quant_fund.models.homotopy_colim import (
    bench_homotopy_colim,
)
from quant_fund.models.periodic_fam import (
    bench_periodic_fam,
)
from quant_fund.models.smash_prod import bench_smash_prod
from quant_fund.models.stable_stem2 import (
    bench_stable_stem2,
)
from quant_fund.models.unstable_tower import (
    bench_unstable_tower,
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


def bench_smash_prod_family(
    seed: int = _SEED + 3752,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "smash_prod",
            bench_smash_prod(seed),
        )
    )


def bench_stable_stem2_family(
    seed: int = _SEED + 3753,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_stem2",
            bench_stable_stem2(seed),
        )
    )


def bench_homotopy_colim_family(
    seed: int = _SEED + 3754,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_colim",
            bench_homotopy_colim(seed),
        )
    )


def bench_unstable_tower_family(
    seed: int = _SEED + 3755,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "unstable_tower",
            bench_unstable_tower(seed),
        )
    )


def bench_periodic_fam_family(
    seed: int = _SEED + 3756,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "periodic_fam",
            bench_periodic_fam(seed),
        )
    )


def bench_chromatic_htpy_family(
    seed: int = _SEED + 3757,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_htpy",
            bench_chromatic_htpy(seed),
        )
    )
