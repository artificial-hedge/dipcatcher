"""Wave-247 adapter tests."""

from quant_fund.research.benches_w247 import (
    bench_atomics_tas_family,
    bench_bakery_lock_family,
    bench_channel_select_family,
    bench_peterson_lock_family,
    bench_rw_lock_family,
    bench_work_stealing_family,
)


def test_bench_atomics_tas_family():
    out = bench_atomics_tas_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_bakery_lock_family():
    out = bench_bakery_lock_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_channel_select_family():
    out = bench_channel_select_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_peterson_lock_family():
    out = bench_peterson_lock_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_rw_lock_family():
    out = bench_rw_lock_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_work_stealing_family():
    out = bench_work_stealing_family()
    assert out and all(k.startswith("synthetic_") for k in out)
