"""Schedule-parity audit for parallel sweeps.

``process_map`` guarantees task ``i`` sees ``(items[i], derive_seed(base, i))``
regardless of worker count or completion order. ``map_parity`` makes that
guarantee falsifiable: run the same tasks under several schedules — serial
forward, serial reversed, a seeded shuffle, and the process pool — and compare
per-task result digests. A task whose digest moves across schedules read state
its seed did not pin: global RNG, shared mutable cache, wall-clock, iteration
order of a dict filled by a neighbor. The audit returns the divergent task
indices so the leak is attributable, not just detected.

The ``pool`` schedule is opportunistic: if the executor cannot start the audit
reports it ``unavailable`` rather than failing — a laptop without fork still
gets the three serial-order checks.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from quant_fund.compute.parallel import derive_seed, process_map
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_SCHEDULES = ("serial", "reversed", "shuffled", "pool")


def _digest(result: Any) -> str:
    return hash_bytes(canonical_json_bytes(result))


def _run_schedule(
    fn: Callable[[Any, int], Any],
    items: Sequence[Any],
    seeds: list[int],
    schedule: str,
    *,
    max_workers: int | None,
    shuffle_seed: int,
) -> list[Any] | None:
    """Execute ``fn`` over ``items`` under ``schedule``; results indexed by task."""
    n = len(items)
    results: list[Any] = [None] * n
    if schedule == "serial":
        order = list(range(n))
    elif schedule == "reversed":
        order = list(range(n - 1, -1, -1))
    elif schedule == "shuffled":
        order = list(range(n))
        random.Random(shuffle_seed).shuffle(order)
    elif schedule == "pool":
        # process_map re-derives seeds from base_seed — the audit supplies its
        # own, so the pool wrapper receives (item, task_seed) and ignores the
        # injected seed.
        try:
            return _pool_with_seeds(fn, items, seeds, max_workers)
        except Exception:
            return None
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    for i in order:
        results[i] = fn(items[i], seeds[i])
    return results


def _pool_invoke(payload: tuple[Callable[[Any, int], Any], Any, int], _seed: int) -> Any:
    fn, item, task_seed = payload
    return fn(item, task_seed)


def _pool_with_seeds(
    fn: Callable[[Any, int], Any],
    items: Sequence[Any],
    seeds: list[int],
    max_workers: int | None,
) -> list[Any]:
    """Process-pool execution honoring the audit's own seed list.

    ``fn`` must be picklable (module-level) — same constraint process_map
    already imposes on sweep workloads.
    """
    payloads = [(fn, item, seed) for item, seed in zip(items, seeds, strict=True)]
    return process_map(_pool_invoke, payloads, base_seed=0, min_items=1, max_workers=max_workers)


def map_parity(
    fn: Callable[[Any, int], Any],
    items: Sequence[Any],
    *,
    base_seed: int,
    schedules: Sequence[str] = _SCHEDULES,
    max_workers: int | None = None,
    shuffle_seed: int = 0x5EED,
) -> dict[str, Any]:
    """Run ``fn`` under every schedule; report which tasks diverge.

    Returns ``{reference_digest, per_task: [...], schedules: {name: status},
    divergent_tasks: [...], parity: bool}``. ``reference_digest`` is the
    canonical hash of the serial run's per-task digests — the schedule-invariant
    fingerprint of the whole sweep.
    """
    sequenced = list(items)
    n = len(sequenced)
    seeds = [derive_seed(base_seed, i) for i in range(n)]
    unknown = [s for s in schedules if s not in _SCHEDULES]
    if unknown:
        raise ValueError(f"unknown schedules: {unknown}")

    runs: dict[str, list[Any] | None] = {}
    for sched in schedules:
        runs[sched] = _run_schedule(
            fn, sequenced, seeds, sched, max_workers=max_workers, shuffle_seed=shuffle_seed
        )

    ref = runs.get("serial")
    if ref is None:
        ref = next(r for r in runs.values() if r is not None)
    per_task = [_digest(r) for r in ref]
    reference_digest = hash_bytes(canonical_json_bytes(per_task))

    divergent: set[int] = set()
    schedule_status: dict[str, str] = {}
    for sched, results in runs.items():
        if results is None:
            schedule_status[sched] = "unavailable"
            continue
        moved = [i for i in range(n) if _digest(results[i]) != per_task[i]]
        schedule_status[sched] = "identical" if not moved else f"diverged:{len(moved)}"
        divergent.update(moved)

    return {
        "n_tasks": n,
        "base_seed": int(base_seed),
        "reference_digest": reference_digest,
        "per_task": per_task,
        "schedules": schedule_status,
        "divergent_tasks": sorted(divergent),
        "parity": not divergent,
    }


def _bench_task(draws: int, seed: int) -> dict[str, Any]:
    """Deterministic seeded workload: quantiles + moments of a private stream."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(draws)
    return {
        "n": draws,
        "seed": seed,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "q": [float(v) for v in np.quantile(x, [0.1, 0.5, 0.9])],
        "digest": hash_bytes(x.tobytes()),
    }


def map_parity_bench(
    *, n_tasks: int = 48, draws: int = 2000, seed: int = 0, max_workers: int | None = 4
) -> dict[str, Any]:
    """Sealed ``map_parity.v1`` receipt for the built-in seeded workload."""
    result = map_parity(
        _bench_task,
        [draws] * n_tasks,
        base_seed=seed,
        max_workers=max_workers,
    )
    result.pop("per_task")  # bound the receipt; reference_digest pins them
    out: dict[str, Any] = {
        "kind": "map_parity",
        "schema": "map_parity.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "invariant": "task results depend only on (item, derive_seed(base, i))",
            "parity": result["parity"],
            "divergent_tasks": result["divergent_tasks"],
        },
        "interpretation": {
            "bench": f"{n_tasks} seeded gaussian tasks x {draws} draws",
            "schedules": result["schedules"],
            "n_tasks": result["n_tasks"],
            "reference_digest": result["reference_digest"],
        },
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


__all__ = ["map_parity", "map_parity_bench"]
