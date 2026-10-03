"""Wave-202 adapters: game-theory canon — mfg_lq, mfg_flocking,
nash_cournot, stackelberg_game, stochastic_game_vi, potential_game —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.mfg_flocking import bench_mfg_flocking
from quant_fund.models.mfg_lq import bench_mfg_lq
from quant_fund.models.nash_cournot import bench_nash_cournot
from quant_fund.models.potential_game import bench_potential_game
from quant_fund.models.stackelberg_game import bench_stackelberg_game
from quant_fund.models.stochastic_game_vi import bench_stochastic_game_vi

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


def bench_stochastic_game_vi_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("stochastic_game_vi", bench_stochastic_game_vi(seed=_SEED + 960))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"stochastic_game_vi bench failed: {exc}") from exc


def bench_mfg_lq_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mfg_lq", bench_mfg_lq(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mfg_lq bench failed: {exc}") from exc


def bench_stackelberg_game_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("stackelberg_game", bench_stackelberg_game(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"stackelberg_game bench failed: {exc}") from exc


def bench_mfg_flocking_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mfg_flocking", bench_mfg_flocking(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mfg_flocking bench failed: {exc}") from exc


def bench_nash_cournot_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nash_cournot", bench_nash_cournot(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nash_cournot bench failed: {exc}") from exc


def bench_potential_game_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("potential_game", bench_potential_game(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"potential_game bench failed: {exc}") from exc
