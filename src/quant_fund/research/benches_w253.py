"""Wave-253 adapters: automata-3 canon — bottom-up tree automata,
Büchi ω-acceptance, tropical WFST, CFG↔PDA, two-way DFA,
register automata — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.buchi_automata import bench_buchi_automata
from quant_fund.models.cfg_pda_equiv import bench_cfg_pda_equiv
from quant_fund.models.register_automata import bench_register_automata
from quant_fund.models.tree_automata import bench_tree_automata
from quant_fund.models.two_way_dfa import bench_two_way_dfa
from quant_fund.models.weighted_fst import bench_weighted_fst

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


def bench_tree_automata_family(seed: int = _SEED + 1300) -> dict[str, float]:
    return bench_tree_automata(seed)


def bench_buchi_automata_family(seed: int = _SEED + 1301) -> dict[str, float]:
    return bench_buchi_automata(seed)


def bench_weighted_fst_family(seed: int = _SEED + 1302) -> dict[str, float]:
    return bench_weighted_fst(seed)


def bench_cfg_pda_equiv_family(seed: int = _SEED + 1303) -> dict[str, float]:
    return bench_cfg_pda_equiv(seed)


def bench_two_way_dfa_family(seed: int = _SEED + 1304) -> dict[str, float]:
    return bench_two_way_dfa(seed)


def bench_register_automata_family(seed: int = _SEED + 1305) -> dict[str, float]:
    return bench_register_automata(seed)
