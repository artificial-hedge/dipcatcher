"""Wave-303 speech/audio codec canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adpcm_ima import bench_adpcm_ima
from quant_fund.models.celp_encode import bench_celp_encode
from quant_fund.models.lpc_analysis import bench_lpc_analysis
from quant_fund.models.mel_cepstrum import bench_mel_cepstrum
from quant_fund.models.mulaw_compand import bench_mulaw_compand
from quant_fund.models.viterbi_vad import bench_viterbi_vad

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


def bench_mulaw_compand_family(seed: int = _SEED + 1724) -> dict[str, float]:
    return _floats(_finite_blob("mulaw_compand", bench_mulaw_compand(seed)))


def bench_adpcm_ima_family(seed: int = _SEED + 1725) -> dict[str, float]:
    return _floats(_finite_blob("adpcm_ima", bench_adpcm_ima(seed)))


def bench_lpc_analysis_family(seed: int = _SEED + 1726) -> dict[str, float]:
    return _floats(_finite_blob("lpc_analysis", bench_lpc_analysis(seed)))


def bench_celp_encode_family(seed: int = _SEED + 1727) -> dict[str, float]:
    return _floats(_finite_blob("celp_encode", bench_celp_encode(seed)))


def bench_mel_cepstrum_family(seed: int = _SEED + 1728) -> dict[str, float]:
    return _floats(_finite_blob("mel_cepstrum", bench_mel_cepstrum(seed)))


def bench_viterbi_vad_family(seed: int = _SEED + 1729) -> dict[str, float]:
    return _floats(_finite_blob("viterbi_vad", bench_viterbi_vad(seed)))
