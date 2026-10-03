"""Wave-256 adapter tests."""

from quant_fund.research.benches_w256 import (
    bench_epoch_reclaim_family,
    bench_flat_combining_family,
    bench_hazard_pointer_family,
    bench_ms_queue_family,
    bench_rcu_lock_family,
    bench_seqlock_family,
)


def test_bench_hazard_pointer_family():
    out = bench_hazard_pointer_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_seqlock_family():
    out = bench_seqlock_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_ms_queue_family():
    out = bench_ms_queue_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_epoch_reclaim_family():
    out = bench_epoch_reclaim_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_flat_combining_family():
    out = bench_flat_combining_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_rcu_lock_family():
    out = bench_rcu_lock_family()
    assert out and all(k.startswith("synthetic_") for k in out)
