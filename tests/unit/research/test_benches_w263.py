"""Wave-263 adapter tests."""

from quant_fund.research.benches_w263 import (
    bench_debounce_fsm_family,
    bench_edf_scheduler_family,
    bench_ring_buffer_family,
    bench_rms_scheduler_family,
    bench_watchdog_task_family,
    bench_wcet_est_family,
)


def test_bench_edf_scheduler_family():
    assert all(k.startswith("synthetic_") for k in bench_edf_scheduler_family())


def test_bench_rms_scheduler_family():
    assert all(k.startswith("synthetic_") for k in bench_rms_scheduler_family())


def test_bench_wcet_est_family():
    assert all(k.startswith("synthetic_") for k in bench_wcet_est_family())


def test_bench_debounce_fsm_family():
    assert all(k.startswith("synthetic_") for k in bench_debounce_fsm_family())


def test_bench_watchdog_task_family():
    assert all(k.startswith("synthetic_") for k in bench_watchdog_task_family())


def test_bench_ring_buffer_family():
    assert all(k.startswith("synthetic_") for k in bench_ring_buffer_family())
