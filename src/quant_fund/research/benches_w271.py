"""Wave-271 networking-3 benches: routing, switching, access, QoS."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.csma_ca import bench_csma_ca
from quant_fund.models.diffserv_qos import bench_diffserv_qos
from quant_fund.models.icmp_path import bench_icmp_path
from quant_fund.models.ospf_lsa import bench_ospf_lsa
from quant_fund.models.stp_spanning import bench_stp_spanning
from quant_fund.models.vlan_tag import bench_vlan_tag

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


def bench_ospf_lsa_family(seed: int = _SEED + 1480) -> dict[str, float]:
    return _floats(_finite_blob("ospf_lsa", bench_ospf_lsa(seed)))


def bench_stp_spanning_family(seed: int = _SEED + 1481) -> dict[str, float]:
    return _floats(_finite_blob("stp_spanning", bench_stp_spanning(seed)))


def bench_vlan_tag_family(seed: int = _SEED + 1482) -> dict[str, float]:
    return _floats(_finite_blob("vlan_tag", bench_vlan_tag(seed)))


def bench_csma_ca_family(seed: int = _SEED + 1483) -> dict[str, float]:
    return _floats(_finite_blob("csma_ca", bench_csma_ca(seed)))


def bench_icmp_path_family(seed: int = _SEED + 1484) -> dict[str, float]:
    return _floats(_finite_blob("icmp_path", bench_icmp_path(seed)))


def bench_diffserv_qos_family(seed: int = _SEED + 1485) -> dict[str, float]:
    return _floats(_finite_blob("diffserv_qos", bench_diffserv_qos(seed)))
