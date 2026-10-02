"""Wave-105 adapters: game-tree search canon — minimax/
alpha-beta + transposition table, UCT Monte Carlo tree
search, AlphaZero-style PUCT, NegaScout/PVS, proof-number
search, and depth-first proof-number (df-pn) — all on a
synthetic subtraction-race DAG with closed-form values.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alpha_beta import bench_alpha_beta
from quant_fund.models.dfpn import bench_dfpn
from quant_fund.models.mcts import bench_mcts
from quant_fund.models.negascout import bench_negascout
from quant_fund.models.proof_number import bench_proof_number
from quant_fund.models.puct import bench_puct

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
                flat[f"{k}_{i}"] = f
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_alpha_beta_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("alpha_beta", bench_alpha_beta(seed=_SEED + 618)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"alpha_beta bench failed: {exc}") from exc


def bench_mcts_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("mcts", bench_mcts(seed=_SEED + 619)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mcts bench failed: {exc}") from exc


def bench_puct_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("puct", bench_puct(seed=_SEED + 620)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"puct bench failed: {exc}") from exc


def bench_negascout_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("negascout", bench_negascout(seed=_SEED + 621)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"negascout bench failed: {exc}") from exc


def bench_proof_number_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("proof_number", bench_proof_number(seed=_SEED + 622))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"proof_number bench failed: {exc}") from exc


def bench_dfpn_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("dfpn", bench_dfpn(seed=_SEED + 623)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dfpn bench failed: {exc}") from exc
