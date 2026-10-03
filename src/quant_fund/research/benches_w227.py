"""Wave-227 adapters: compression canon — huffman_codes,
arithmetic_coding, lzw_compress, golomb_rice, rans_coder, lz78_dict —
benched on SYNTHETIC corpora. Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arithmetic_coding import bench_arithmetic_coding
from quant_fund.models.golomb_rice import bench_golomb_rice
from quant_fund.models.huffman_codes import bench_huffman_codes
from quant_fund.models.lz78_dict import bench_lz78_dict
from quant_fund.models.lzw_compress import bench_lzw_compress
from quant_fund.models.rans_coder import bench_rans_coder

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


def bench_arithmetic_coding_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("arithmetic_coding", bench_arithmetic_coding(seed=_SEED + 1040))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"arithmetic_coding bench failed: {exc}") from exc


def bench_golomb_rice_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("golomb_rice", bench_golomb_rice(seed=_SEED + 1041)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"golomb_rice bench failed: {exc}") from exc


def bench_huffman_codes_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("huffman_codes", bench_huffman_codes(seed=_SEED + 1042)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"huffman_codes bench failed: {exc}") from exc


def bench_lz78_dict_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lz78_dict", bench_lz78_dict(seed=_SEED + 1043)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lz78_dict bench failed: {exc}") from exc


def bench_lzw_compress_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lzw_compress", bench_lzw_compress(seed=_SEED + 1044)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lzw_compress bench failed: {exc}") from exc


def bench_rans_coder_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rans_coder", bench_rans_coder(seed=_SEED + 1045)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rans_coder bench failed: {exc}") from exc
