"""Wave-249 adapters: bioinformatics canon — Smith–Waterman,
Needleman–Wunsch affine, de Bruijn assembly, FM-index, UPGMA,
PSSM motif scan — SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.debruijn_assemble import bench_debruijn_assemble
from quant_fund.models.fm_index import bench_fm_index
from quant_fund.models.motif_scan import bench_motif_scan
from quant_fund.models.needleman_wunsch import bench_needleman_wunsch
from quant_fund.models.smith_waterman import bench_smith_waterman
from quant_fund.models.upgma_tree import bench_upgma_tree

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


def bench_debruijn_assemble_family(seed: int = _SEED + 1260) -> dict[str, float]:
    return bench_debruijn_assemble(seed)


def bench_fm_index_family(seed: int = _SEED + 1261) -> dict[str, float]:
    return bench_fm_index(seed)


def bench_motif_scan_family(seed: int = _SEED + 1262) -> dict[str, float]:
    return bench_motif_scan(seed)


def bench_needleman_wunsch_family(seed: int = _SEED + 1263) -> dict[str, float]:
    return bench_needleman_wunsch(seed)


def bench_smith_waterman_family(seed: int = _SEED + 1264) -> dict[str, float]:
    return bench_smith_waterman(seed)


def bench_upgma_tree_family(seed: int = _SEED + 1265) -> dict[str, float]:
    return bench_upgma_tree(seed)
