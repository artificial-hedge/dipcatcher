"""Wave-222 adapters: number-theory canon — miller_rabin, pollard_rho,
tonelli_shanks, continued_fraction, crt_garner, ec_scalar —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.continued_fraction import bench_continued_fraction
from quant_fund.models.crt_garner import bench_crt_garner
from quant_fund.models.ec_scalar import bench_ec_scalar
from quant_fund.models.miller_rabin import bench_miller_rabin
from quant_fund.models.pollard_rho import bench_pollard_rho
from quant_fund.models.tonelli_shanks import bench_tonelli_shanks

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


def bench_continued_fraction_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("continued_fraction", bench_continued_fraction(seed=_SEED + 990))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"continued_fraction bench failed: {exc}") from exc


def bench_crt_garner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("crt_garner", bench_crt_garner(seed=_SEED + 991)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crt_garner bench failed: {exc}") from exc


def bench_ec_scalar_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ec_scalar", bench_ec_scalar(seed=_SEED + 992)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ec_scalar bench failed: {exc}") from exc


def bench_miller_rabin_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("miller_rabin", bench_miller_rabin(seed=_SEED + 993)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"miller_rabin bench failed: {exc}") from exc


def bench_pollard_rho_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pollard_rho", bench_pollard_rho(seed=_SEED + 994)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pollard_rho bench failed: {exc}") from exc


def bench_tonelli_shanks_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tonelli_shanks", bench_tonelli_shanks(seed=_SEED + 995)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tonelli_shanks bench failed: {exc}") from exc
