"""Wave-265 program-analysis canon adapter: SYNTHETIC benches."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.asan_shadow import bench_asan_shadow
from quant_fund.models.contract_check import bench_contract_check
from quant_fund.models.fuzzer_mutate import bench_fuzzer_mutate
from quant_fund.models.grammar_fuzz import bench_grammar_fuzz
from quant_fund.models.symbolic_exec import bench_symbolic_exec
from quant_fund.models.taint_track import bench_taint_track

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


def bench_fuzzer_mutate_family(seed: int = _SEED + 1420) -> dict[str, float]:
    return bench_fuzzer_mutate(seed)


def bench_taint_track_family(seed: int = _SEED + 1421) -> dict[str, float]:
    return bench_taint_track(seed)


def bench_asan_shadow_family(seed: int = _SEED + 1422) -> dict[str, float]:
    return bench_asan_shadow(seed)


def bench_symbolic_exec_family(seed: int = _SEED + 1423) -> dict[str, float]:
    return bench_symbolic_exec(seed)


def bench_contract_check_family(seed: int = _SEED + 1424) -> dict[str, float]:
    return bench_contract_check(seed)


def bench_grammar_fuzz_family(seed: int = _SEED + 1425) -> dict[str, float]:
    return bench_grammar_fuzz(seed)
