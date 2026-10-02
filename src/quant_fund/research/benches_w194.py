"""Wave-126 adapters: exec-summary stochastic-process sim canon — levy_jump,
cir_sim, hawkes_thinning, ou_bridge, poisson_thinning, gp_bridge —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cir_sim import bench_cir_sim
from quant_fund.models.gp_bridge import bench_gp_bridge
from quant_fund.models.hawkes_thinning import bench_hawkes_thinning
from quant_fund.models.levy_jump import bench_levy_jump
from quant_fund.models.ou_bridge import bench_ou_bridge
from quant_fund.models.poisson_thinning import bench_poisson_thinning

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_levy_jump_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("levy_jump", bench_levy_jump(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"levy_jump bench failed: {exc}") from exc


def bench_cir_sim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cir_sim", bench_cir_sim(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cir_sim bench failed: {exc}") from exc


def bench_hawkes_thinning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hawkes_thinning", bench_hawkes_thinning(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hawkes_thinning bench failed: {exc}") from exc


def bench_ou_bridge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ou_bridge", bench_ou_bridge(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ou_bridge bench failed: {exc}") from exc


def bench_poisson_thinning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("poisson_thinning", bench_poisson_thinning(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"poisson_thinning bench failed: {exc}") from exc


def bench_gp_bridge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gp_bridge", bench_gp_bridge(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gp_bridge bench failed: {exc}") from exc
