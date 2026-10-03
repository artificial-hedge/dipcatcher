"""Wave-203 adapters: information-geometry canon — cwt_ridge,
cepstrum_pitch, mvdr_beamformer, hilbert_instant, lpc_formant, goertzel_detect —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cepstrum_pitch import bench_cepstrum_pitch
from quant_fund.models.cwt_ridge import bench_cwt_ridge
from quant_fund.models.goertzel_detect import bench_goertzel_detect
from quant_fund.models.hilbert_instant import bench_hilbert_instant
from quant_fund.models.lpc_formant import bench_lpc_formant
from quant_fund.models.mvdr_beamformer import bench_mvdr_beamformer

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


def bench_lpc_formant_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lpc_formant", bench_lpc_formant(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lpc_formant bench failed: {exc}") from exc


def bench_cwt_ridge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cwt_ridge", bench_cwt_ridge(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cwt_ridge bench failed: {exc}") from exc


def bench_hilbert_instant_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hilbert_instant", bench_hilbert_instant(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hilbert_instant bench failed: {exc}") from exc


def bench_cepstrum_pitch_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cepstrum_pitch", bench_cepstrum_pitch(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cepstrum_pitch bench failed: {exc}") from exc


def bench_mvdr_beamformer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mvdr_beamformer", bench_mvdr_beamformer(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mvdr_beamformer bench failed: {exc}") from exc


def bench_goertzel_detect_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("goertzel_detect", bench_goertzel_detect(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"goertzel_detect bench failed: {exc}") from exc
