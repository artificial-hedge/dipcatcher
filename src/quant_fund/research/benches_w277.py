"""Wave-277 signal-processing-4 benches: transforms + resampling."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chirp_z import bench_chirp_z
from quant_fund.models.decimate_int import bench_decimate_int
from quant_fund.models.fir_window import bench_fir_window
from quant_fund.models.prony_model import bench_prony_model
from quant_fund.models.stft_istft import bench_stft_istft
from quant_fund.models.wola_synth import bench_wola_synth

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


def bench_stft_istft_family(seed: int = _SEED + 1540) -> dict[str, float]:
    return _floats(_finite_blob("stft_istft", bench_stft_istft(seed)))


def bench_chirp_z_family(seed: int = _SEED + 1541) -> dict[str, float]:
    return _floats(_finite_blob("chirp_z", bench_chirp_z(seed)))


def bench_fir_window_family(seed: int = _SEED + 1542) -> dict[str, float]:
    return _floats(_finite_blob("fir_window", bench_fir_window(seed)))


def bench_prony_model_family(seed: int = _SEED + 1543) -> dict[str, float]:
    return _floats(_finite_blob("prony_model", bench_prony_model(seed)))


def bench_wola_synth_family(seed: int = _SEED + 1544) -> dict[str, float]:
    return _floats(_finite_blob("wola_synth", bench_wola_synth(seed)))


def bench_decimate_int_family(seed: int = _SEED + 1545) -> dict[str, float]:
    return _floats(_finite_blob("decimate_int", bench_decimate_int(seed)))
