"""Tests for quant_fund.compute.map_parity."""

from __future__ import annotations

import random

import pytest

from quant_fund.compute.map_parity import map_parity, map_parity_bench
from quant_fund.compute.parallel import derive_seed


def _seeded_task(item: int, seed: int) -> dict[str, int]:
    return {"item": item, "value": seed % 97 + item}


def test_serial_schedules_identical():
    out = map_parity(
        _seeded_task,
        list(range(40)),
        base_seed=7,
        schedules=("serial", "reversed", "shuffled"),
    )
    assert out["parity"]
    assert out["divergent_tasks"] == []
    assert all(v == "identical" for v in out["schedules"].values())


def test_leaky_global_rng_flagged():
    stateful = random.Random(1234)

    def leaky(item: int, seed: int) -> dict[str, int]:
        return {"item": item, "roll": stateful.randint(0, 10**9)}

    out = map_parity(
        leaky,
        list(range(24)),
        base_seed=1,
        schedules=("serial", "reversed"),
    )
    assert not out["parity"]
    # reversed order moves every task's draw except possibly the last
    assert len(out["divergent_tasks"]) >= 23


def test_pool_schedule_matches_serial():
    out = map_parity(
        _seeded_task,
        list(range(16)),
        base_seed=3,
        schedules=("serial", "pool"),
        max_workers=2,
    )
    assert out["schedules"]["pool"] in ("identical", "unavailable")
    if out["schedules"]["pool"] == "identical":
        assert out["parity"]


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="unknown schedules"):
        map_parity(_seeded_task, [1], base_seed=0, schedules=("bogus",))


def test_bench_sealed_and_deterministic():
    r1 = map_parity_bench(n_tasks=12, draws=500, seed=1, max_workers=2)
    r2 = map_parity_bench(n_tasks=12, draws=500, seed=1, max_workers=2)
    assert r1["schema"] == "map_parity.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["claim"]["parity"] is True
    assert r1 == r2


def test_derive_seed_is_per_index():
    seeds = {derive_seed(0, i) for i in range(64)}
    assert len(seeds) == 64  # no collisions in the audit range
