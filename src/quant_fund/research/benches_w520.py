"""Wave-520 automorphic-GL(n) bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.converse_thm import bench_converse_thm
from quant_fund.models.gln_automorphic import bench_gln_automorphic
from quant_fund.models.godement_jacq import bench_godement_jacq
from quant_fund.models.langlands_lfunc import bench_langlands_lfunc
from quant_fund.models.rankin_selberg import bench_rankin_selberg
from quant_fund.models.whittaker_model import bench_whittaker_model

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


def bench_gln_automorphic_family(seed: int = _SEED + 3026) -> dict[str, float]:
    return _floats(_finite_blob("gln_automorphic", bench_gln_automorphic(seed)))


def bench_whittaker_model_family(seed: int = _SEED + 3027) -> dict[str, float]:
    return _floats(_finite_blob("whittaker_model", bench_whittaker_model(seed)))


def bench_godement_jacq_family(seed: int = _SEED + 3028) -> dict[str, float]:
    return _floats(_finite_blob("godement_jacq", bench_godement_jacq(seed)))


def bench_rankin_selberg_family(seed: int = _SEED + 3029) -> dict[str, float]:
    return _floats(_finite_blob("rankin_selberg", bench_rankin_selberg(seed)))


def bench_langlands_lfunc_family(seed: int = _SEED + 3030) -> dict[str, float]:
    return _floats(_finite_blob("langlands_lfunc", bench_langlands_lfunc(seed)))


def bench_converse_thm_family(seed: int = _SEED + 3031) -> dict[str, float]:
    return _floats(_finite_blob("converse_thm", bench_converse_thm(seed)))
