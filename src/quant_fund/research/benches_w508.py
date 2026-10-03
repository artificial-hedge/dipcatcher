"""Wave-508 anabelian-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anabelian_geo import bench_anabelian_geo
from quant_fund.models.etale_pi1 import bench_etale_pi1
from quant_fund.models.fundamental_grp import bench_fundamental_grp
from quant_fund.models.groth_tei import bench_groth_tei
from quant_fund.models.section_conj import bench_section_conj
from quant_fund.models.tamagawa_mochi import bench_tamagawa_mochi

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


def bench_anabelian_geo_family(seed: int = _SEED + 2954) -> dict[str, float]:
    return _floats(_finite_blob("anabelian_geo", bench_anabelian_geo(seed)))


def bench_section_conj_family(seed: int = _SEED + 2955) -> dict[str, float]:
    return _floats(_finite_blob("section_conj", bench_section_conj(seed)))


def bench_fundamental_grp_family(seed: int = _SEED + 2956) -> dict[str, float]:
    return _floats(_finite_blob("fundamental_grp", bench_fundamental_grp(seed)))


def bench_etale_pi1_family(seed: int = _SEED + 2957) -> dict[str, float]:
    return _floats(_finite_blob("etale_pi1", bench_etale_pi1(seed)))


def bench_groth_tei_family(seed: int = _SEED + 2958) -> dict[str, float]:
    return _floats(_finite_blob("groth_tei", bench_groth_tei(seed)))


def bench_tamagawa_mochi_family(seed: int = _SEED + 2959) -> dict[str, float]:
    return _floats(_finite_blob("tamagawa_mochi", bench_tamagawa_mochi(seed)))
