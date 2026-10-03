"""Wave-245 adapters: OS-2 canon — ELF loader, free-list malloc,
demand pager, MLFQ scheduler, semaphore monitor, syscall layer —
SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.elf_loader import bench_elf_loader
from quant_fund.models.malloc_freelist import bench_malloc_freelist
from quant_fund.models.mlfq_sched import bench_mlfq_sched
from quant_fund.models.mmap_pager import bench_mmap_pager
from quant_fund.models.semaphore_monitor import bench_semaphore_monitor
from quant_fund.models.syscall_layer import bench_syscall_layer

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


def bench_elf_loader_family(seed: int = _SEED + 1220) -> dict[str, float]:
    return bench_elf_loader(seed)


def bench_malloc_freelist_family(seed: int = _SEED + 1221) -> dict[str, float]:
    return bench_malloc_freelist(seed)


def bench_mlfq_sched_family(seed: int = _SEED + 1222) -> dict[str, float]:
    return bench_mlfq_sched(seed)


def bench_mmap_pager_family(seed: int = _SEED + 1223) -> dict[str, float]:
    return bench_mmap_pager(seed)


def bench_semaphore_monitor_family(seed: int = _SEED + 1224) -> dict[str, float]:
    return bench_semaphore_monitor(seed)


def bench_syscall_layer_family(seed: int = _SEED + 1225) -> dict[str, float]:
    return bench_syscall_layer(seed)
