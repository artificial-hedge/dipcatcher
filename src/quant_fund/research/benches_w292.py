"""Wave-292 networking-4 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.doh_wire import bench_doh_wire
from quant_fund.models.qpack_pack import bench_qpack_pack
from quant_fund.models.quic_streams import bench_quic_streams
from quant_fund.models.sctp_tsn import bench_sctp_tsn
from quant_fund.models.tls13_trans import bench_tls13_trans
from quant_fund.models.wg_ik import bench_wg_ik

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


def bench_quic_streams_family(seed: int = _SEED + 1658) -> dict[str, float]:
    return _floats(_finite_blob("quic_streams", bench_quic_streams(seed)))


def bench_tls13_trans_family(seed: int = _SEED + 1659) -> dict[str, float]:
    return _floats(_finite_blob("tls13_trans", bench_tls13_trans(seed)))


def bench_qpack_pack_family(seed: int = _SEED + 1660) -> dict[str, float]:
    return _floats(_finite_blob("qpack_pack", bench_qpack_pack(seed)))


def bench_wg_ik_family(seed: int = _SEED + 1661) -> dict[str, float]:
    return _floats(_finite_blob("wg_ik", bench_wg_ik(seed)))


def bench_doh_wire_family(seed: int = _SEED + 1662) -> dict[str, float]:
    return _floats(_finite_blob("doh_wire", bench_doh_wire(seed)))


def bench_sctp_tsn_family(seed: int = _SEED + 1663) -> dict[str, float]:
    return _floats(_finite_blob("sctp_tsn", bench_sctp_tsn(seed)))
