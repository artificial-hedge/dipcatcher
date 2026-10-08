"""Unit tests for quant_fund.models.cuckoo_filter."""

from __future__ import annotations

import random

from quant_fund.models.cuckoo_filter import Cuckoo


def _fill(cf: Cuckoo, n_items: int) -> list[int]:
    rng = random.Random(1234)
    for _ in range(n_items):
        cf.add(rng.randrange(10**9))
    return list(cf.tab)


def test_kick_path_does_not_consume_global_rng() -> None:
    """The eviction coin flip must come from the filter's own RNG —
    before the fix, kick draws pulled from global random state, so
    interleaving a cuckoo insert changed downstream random draws."""
    random.seed(9)
    expected = [random.random() for _ in range(4)]
    random.seed(9)
    # small table + many inserts forces kick-path evictions
    _fill(Cuckoo(16, rng=random.Random(0)), 24)
    assert [random.random() for _ in range(4)] == expected


def test_kick_path_deterministic_for_same_seed() -> None:
    items_rng = random.Random(7)
    items = [items_rng.randrange(10**9) for _ in range(24)]
    random.seed(11)
    t1 = list(_fill(Cuckoo(16, rng=random.Random(5)), 0))
    cf1 = Cuckoo(16, rng=random.Random(5))
    for x in items:
        cf1.add(x)
    random.seed(999)  # different global state must not matter
    cf2 = Cuckoo(16, rng=random.Random(5))
    for x in items:
        cf2.add(x)
    assert cf1.tab == cf2.tab == t1 or cf1.tab == cf2.tab


def test_default_rng_deterministic() -> None:
    items = [random.Random(3).randrange(10**9) for _ in range(24)]
    random.seed(1)
    cf1 = Cuckoo(16)
    for x in items:
        cf1.add(x)
    random.seed(777)
    cf2 = Cuckoo(16)
    for x in items:
        cf2.add(x)
    assert cf1.tab == cf2.tab
