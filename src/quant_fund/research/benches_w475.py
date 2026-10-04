"""Wave-475 homotopical-algebra bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chiral_alg import bench_chiral_alg
from quant_fund.models.cyclic_hk import bench_cyclic_hk
from quant_fund.models.dendroidal import bench_dendroidal
from quant_fund.models.infty_operad import bench_infty_operad
from quant_fund.models.seq_spectra import bench_seq_spectra
from quant_fund.models.sifted_cat import bench_sifted_cat

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


def bench_dendroidal_family(seed: int = _SEED + 2756) -> dict[str, float]:
    return _floats(_finite_blob("dendroidal", bench_dendroidal(seed)))


def bench_infty_operad_family(seed: int = _SEED + 2757) -> dict[str, float]:
    return _floats(_finite_blob("infty_operad", bench_infty_operad(seed)))


def bench_cyclic_hk_family(seed: int = _SEED + 2758) -> dict[str, float]:
    return _floats(_finite_blob("cyclic_hk", bench_cyclic_hk(seed)))


def bench_chiral_alg_family(seed: int = _SEED + 2759) -> dict[str, float]:
    return _floats(_finite_blob("chiral_alg", bench_chiral_alg(seed)))


def bench_sifted_cat_family(seed: int = _SEED + 2760) -> dict[str, float]:
    return _floats(_finite_blob("sifted_cat", bench_sifted_cat(seed)))


def bench_seq_spectra_family(seed: int = _SEED + 2761) -> dict[str, float]:
    return _floats(_finite_blob("seq_spectra", bench_seq_spectra(seed)))
