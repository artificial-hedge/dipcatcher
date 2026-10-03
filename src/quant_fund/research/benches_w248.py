"""Wave-248 adapters: formal-language canon — Turing machine, PDA,
Brzozowski derivatives, DFA equivalence/minimization, Mealy↔Moore,
cellular automata — SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brzozowski_deriv import bench_brzozowski_deriv
from quant_fund.models.cellular_automata import bench_cellular_automata
from quant_fund.models.dfa_equiv import bench_dfa_equiv
from quant_fund.models.mealy_moore import bench_mealy_moore
from quant_fund.models.pda_sim import bench_pda_sim
from quant_fund.models.turing_machine import bench_turing_machine

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


def bench_brzozowski_deriv_family(seed: int = _SEED + 1250) -> dict[str, float]:
    return bench_brzozowski_deriv(seed)


def bench_cellular_automata_family(seed: int = _SEED + 1251) -> dict[str, float]:
    return bench_cellular_automata(seed)


def bench_dfa_equiv_family(seed: int = _SEED + 1252) -> dict[str, float]:
    return bench_dfa_equiv(seed)


def bench_mealy_moore_family(seed: int = _SEED + 1253) -> dict[str, float]:
    return bench_mealy_moore(seed)


def bench_pda_sim_family(seed: int = _SEED + 1254) -> dict[str, float]:
    return bench_pda_sim(seed)


def bench_turing_machine_family(seed: int = _SEED + 1255) -> dict[str, float]:
    return bench_turing_machine(seed)
