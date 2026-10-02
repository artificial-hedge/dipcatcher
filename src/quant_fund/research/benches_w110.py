"""Wave-110 adapters: digital-comms canon — convolutional
encode + hard/soft Viterbi, GF(256) arithmetic, Reed–Solomon
BM/Chien/magnitude decode, Costas carrier recovery, Gardner
timing recovery, RRC matched filtering — each benched on
SYNTHETIC channels with known references. Adapters flatten to
a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.costas import bench_costas
from quant_fund.models.gardner import bench_gardner
from quant_fund.models.gf256 import bench_gf256
from quant_fund.models.reed_solomon import bench_reed_solomon
from quant_fund.models.rrc_filter import bench_rrc_filter
from quant_fund.models.viterbi_decode import bench_viterbi_decode

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


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_viterbi_decode_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("viterbi_decode", bench_viterbi_decode(seed=_SEED + 648))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"viterbi_decode bench failed: {exc}") from exc


def bench_gf256_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("gf256", bench_gf256(seed=_SEED + 649)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gf256 bench failed: {exc}") from exc


def bench_reed_solomon_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("reed_solomon", bench_reed_solomon(seed=_SEED + 650))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reed_solomon bench failed: {exc}") from exc


def bench_costas_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("costas", bench_costas(seed=_SEED + 651)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"costas bench failed: {exc}") from exc


def bench_gardner_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("gardner", bench_gardner(seed=_SEED + 652)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gardner bench failed: {exc}") from exc


def bench_rrc_filter_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("rrc_filter", bench_rrc_filter(seed=_SEED + 653)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rrc_filter bench failed: {exc}") from exc
