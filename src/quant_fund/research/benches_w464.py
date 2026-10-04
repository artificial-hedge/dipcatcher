"""Wave-464 chromatic-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chromatic_conv import bench_chromatic_conv
from quant_fund.models.k_n_local import bench_k_n_local
from quant_fund.models.morava_e import bench_morava_e
from quant_fund.models.nilpotence_dev import bench_nilpotence_dev
from quant_fund.models.telescopic import bench_telescopic
from quant_fund.models.tmf_spectrum import bench_tmf_spectrum

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


def bench_morava_e_family(seed: int = _SEED + 2690) -> dict[str, float]:
    return _floats(_finite_blob("morava_e", bench_morava_e(seed)))


def bench_tmf_spectrum_family(seed: int = _SEED + 2691) -> dict[str, float]:
    return _floats(_finite_blob("tmf_spectrum", bench_tmf_spectrum(seed)))


def bench_k_n_local_family(seed: int = _SEED + 2692) -> dict[str, float]:
    return _floats(_finite_blob("k_n_local", bench_k_n_local(seed)))


def bench_chromatic_conv_family(seed: int = _SEED + 2693) -> dict[str, float]:
    return _floats(_finite_blob("chromatic_conv", bench_chromatic_conv(seed)))


def bench_nilpotence_dev_family(seed: int = _SEED + 2694) -> dict[str, float]:
    return _floats(_finite_blob("nilpotence_dev", bench_nilpotence_dev(seed)))


def bench_telescopic_family(seed: int = _SEED + 2695) -> dict[str, float]:
    return _floats(_finite_blob("telescopic", bench_telescopic(seed)))
