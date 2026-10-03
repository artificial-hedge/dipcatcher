"""Wave-210 adapters: coding-theory canon — ldpc_decoder,
turbo_decoder, polar_code, bch_code, crc_check, conv_interleaver —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bch_code import bench_bch_code
from quant_fund.models.conv_interleaver import bench_conv_interleaver
from quant_fund.models.crc_check import bench_crc_check
from quant_fund.models.ldpc_decoder import bench_ldpc_decoder
from quant_fund.models.polar_code import bench_polar_code
from quant_fund.models.turbo_decoder import bench_turbo_decoder

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


def bench_crc_check_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("crc_check", bench_crc_check(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crc_check bench failed: {exc}") from exc


def bench_ldpc_decoder_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ldpc_decoder", bench_ldpc_decoder(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ldpc_decoder bench failed: {exc}") from exc


def bench_bch_code_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bch_code", bench_bch_code(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bch_code bench failed: {exc}") from exc


def bench_turbo_decoder_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("turbo_decoder", bench_turbo_decoder(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"turbo_decoder bench failed: {exc}") from exc


def bench_polar_code_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("polar_code", bench_polar_code(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"polar_code bench failed: {exc}") from exc


def bench_conv_interleaver_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("conv_interleaver", bench_conv_interleaver(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"conv_interleaver bench failed: {exc}") from exc
