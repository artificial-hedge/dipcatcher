"""Wave-259 adapters: networking-2 canon — BGP path-vector,
DNS resolver+cache, NAT hole-punch, ARP, DHCP lease,
learning switch — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arp_table import bench_arp_table
from quant_fund.models.bgp_pathvec import bench_bgp_pathvec
from quant_fund.models.dhcp_lease import bench_dhcp_lease
from quant_fund.models.dns_resolver import bench_dns_resolver
from quant_fund.models.eth_switch import bench_eth_switch
from quant_fund.models.nat_traversal import bench_nat_traversal

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


def bench_bgp_pathvec_family(seed: int = _SEED + 1360) -> dict[str, float]:
    return bench_bgp_pathvec(seed)


def bench_dns_resolver_family(seed: int = _SEED + 1361) -> dict[str, float]:
    return bench_dns_resolver(seed)


def bench_nat_traversal_family(seed: int = _SEED + 1362) -> dict[str, float]:
    return bench_nat_traversal(seed)


def bench_arp_table_family(seed: int = _SEED + 1363) -> dict[str, float]:
    return bench_arp_table(seed)


def bench_dhcp_lease_family(seed: int = _SEED + 1364) -> dict[str, float]:
    return bench_dhcp_lease(seed)


def bench_eth_switch_family(seed: int = _SEED + 1365) -> dict[str, float]:
    return bench_eth_switch(seed)
