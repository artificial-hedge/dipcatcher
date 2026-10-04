"""Wave-803 signature bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.expected_sig import (
    bench_expected_sig,
)
from quant_fund.models.pde_signature import (
    bench_pde_signature,
)
from quant_fund.models.sig_inversion import (
    bench_sig_inversion,
)
from quant_fund.models.signature_gan2 import (
    bench_signature_gan2,
)
from quant_fund.models.signature_kernel import (
    bench_signature_kernel,
)
from quant_fund.models.truncated_sig import (
    bench_truncated_sig,
)

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


def bench_signature_kernel_family(
    seed: int = _SEED + 19200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "signature_kernel",
            bench_signature_kernel(seed),
        )
    )


def bench_pde_signature_family(
    seed: int = _SEED + 19201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pde_signature",
            bench_pde_signature(seed),
        )
    )


def bench_truncated_sig_family(
    seed: int = _SEED + 19202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "truncated_sig",
            bench_truncated_sig(seed),
        )
    )


def bench_signature_gan2_family(
    seed: int = _SEED + 19203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "signature_gan2",
            bench_signature_gan2(seed),
        )
    )


def bench_expected_sig_family(
    seed: int = _SEED + 19204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "expected_sig",
            bench_expected_sig(seed),
        )
    )


def bench_sig_inversion_family(
    seed: int = _SEED + 19205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sig_inversion",
            bench_sig_inversion(seed),
        )
    )
