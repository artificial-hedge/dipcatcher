"""FitCache bounds and key semantics — the fail-closed edges.

A cache may only ever return what was stored under exactly that content key.
These probes cover the constructor guard, LRU eviction order, and the ways two
arrays can be "logically equal" but stored under different bytes.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.compute.cache import FitCache


@pytest.mark.parametrize("max_entries", [0, -1, -10])
def test_non_positive_capacity_is_rejected(max_entries: int) -> None:
    with pytest.raises(ValueError, match="max_entries must be a positive integer"):
        FitCache(max_entries=max_entries)


def test_capacity_one_keeps_only_the_newest_entry() -> None:
    cache = FitCache(max_entries=1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_eviction_is_lru_not_fifo() -> None:
    cache = FitCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # refresh a, so b is now the oldest
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_overwrite_refreshes_recency_and_value() -> None:
    cache = FitCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 10


def test_a_miss_does_not_create_an_entry() -> None:
    cache = FitCache(max_entries=4)
    assert cache.get("missing") is None
    assert len(cache._store) == 0  # noqa: SLF001


def test_clear_drops_every_entry() -> None:
    cache = FitCache(max_entries=4)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert len(cache._store) == 0  # noqa: SLF001


def test_stored_none_is_indistinguishable_from_a_miss_by_design() -> None:
    cache = FitCache(max_entries=2)
    cache.put("k", None)
    assert cache.get("k") is None


def test_get_returns_the_identical_object() -> None:
    payload = {"weights": [0.1, 0.2], "nested": {"rows": 3}}
    cache = FitCache(max_entries=2)
    cache.put("k", payload)
    assert cache.get("k") is payload


def test_key_separates_tag_shape_and_dtype() -> None:
    cache = FitCache()
    base = np.arange(8.0)
    key = cache.hash_key(("m",), (base,))
    assert cache.hash_key(("m2",), (base,)) != key
    assert cache.hash_key(("m",), (base.reshape(2, 4),)) != key
    assert cache.hash_key(("m",), (base.astype(np.int64),)) != key
    assert cache.hash_key(("m",), (base.copy(),)) == key


def test_single_element_mutation_never_collides() -> None:
    cache = FitCache()
    base = np.linspace(0.0, 1.0, 16)
    key = cache.hash_key(("m",), (base,))
    mutated = base.copy()
    mutated[-1] = np.nextafter(mutated[-1], np.inf)
    assert cache.hash_key(("m",), (mutated,)) != key


def test_empty_and_scalar_arrays_hash_without_raising() -> None:
    cache = FitCache()
    assert cache.hash_key((), ()) != cache.hash_key((), (np.array([]),))
    assert cache.hash_key(("s",), (np.array(5.0),)) != cache.hash_key(
        ("s",), (np.array(5.0, dtype=np.float32),)
    )


def test_nan_payloads_hash_deterministically() -> None:
    cache = FitCache()
    payload = np.array([np.nan, 1.0])
    assert cache.hash_key(("m",), (payload,)) == cache.hash_key(("m",), (payload.copy(),))
