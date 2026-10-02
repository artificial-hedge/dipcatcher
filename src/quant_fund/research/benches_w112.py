"""Wave-112 adapters: DSP filter-design canon — Parks–McClellan
equiripple FIR, analog-prototype IIR (Butterworth/Chebyshev-I) via
prewarped bilinear transform, RBJ biquads + SOS cascade, zero-phase
filtfilt, polyphase rational resampling, and the Farrow variable
fractional-delay structure — each benched on SYNTHETIC signals with
spec assertions. Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.biquad import bench_biquad
from quant_fund.models.farrow import bench_farrow
from quant_fund.models.filtfilt import bench_filtfilt
from quant_fund.models.iir_design import bench_iir_design
from quant_fund.models.remez import bench_remez
from quant_fund.models.resample_poly import bench_resample_poly

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


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_remez_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("remez", bench_remez(seed=_SEED + 660)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"remez bench failed: {exc}") from exc


def bench_iir_design_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("iir_design", bench_iir_design(seed=_SEED + 661)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"iir_design bench failed: {exc}") from exc


def bench_biquad_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("biquad", bench_biquad(seed=_SEED + 662)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"biquad bench failed: {exc}") from exc


def bench_filtfilt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("filtfilt", bench_filtfilt(seed=_SEED + 663)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"filtfilt bench failed: {exc}") from exc


def bench_resample_poly_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("resample_poly", bench_resample_poly(seed=_SEED + 664)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"resample_poly bench failed: {exc}") from exc


def bench_farrow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("farrow", bench_farrow(seed=_SEED + 665)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"farrow bench failed: {exc}") from exc
