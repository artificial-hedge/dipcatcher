"""Wave-243 adapters: numeric-2 canon — radix-2 FFT, NTT,
Karatsuba, Strassen, integer sqrt, bignum — SYNTHETIC correctness
benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bignum import bench_bignum
from quant_fund.models.fft_radix2 import bench_fft_radix2
from quant_fund.models.int_sqrt import bench_int_sqrt
from quant_fund.models.karatsuba import bench_karatsuba
from quant_fund.models.ntt import bench_ntt
from quant_fund.models.strassen import bench_strassen

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


def bench_bignum_family(seed: int = _SEED + 1200) -> dict[str, float]:
    return bench_bignum(seed)


def bench_fft_radix2_family(seed: int = _SEED + 1201) -> dict[str, float]:
    return bench_fft_radix2(seed)


def bench_int_sqrt_family(seed: int = _SEED + 1202) -> dict[str, float]:
    return bench_int_sqrt(seed)


def bench_karatsuba_family(seed: int = _SEED + 1203) -> dict[str, float]:
    return bench_karatsuba(seed)


def bench_ntt_family(seed: int = _SEED + 1204) -> dict[str, float]:
    return bench_ntt(seed)


def bench_strassen_family(seed: int = _SEED + 1205) -> dict[str, float]:
    return bench_strassen(seed)
