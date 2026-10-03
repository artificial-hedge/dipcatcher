"""Tests for quant_fund.compute.cache_audit."""

from __future__ import annotations

import numpy as np

from quant_fund.compute.cache import FitCache
from quant_fund.compute.cache_audit import cache_audit, cache_audit_bench


def test_audit_clean():
    r = cache_audit()
    assert r["n_violations"] == 0
    assert r["verdict"] == "ok"


def test_mutation_invariant_directly():
    cache = FitCache()
    a = np.arange(16.0)
    k1 = cache.hash_key(("t",), (a,))
    a[0] += 1
    assert cache.hash_key(("t",), (a,)) != k1


def test_bench_sealed():
    r = cache_audit_bench()
    assert r["schema"] == "cache_audit.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r == cache_audit_bench()
