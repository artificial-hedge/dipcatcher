"""Wave-263 real-time unit tests."""

from quant_fund.models.debounce_fsm import bench_debounce_fsm
from quant_fund.models.edf_scheduler import bench_edf_scheduler
from quant_fund.models.ring_buffer import bench_ring_buffer
from quant_fund.models.rms_scheduler import bench_rms_scheduler
from quant_fund.models.watchdog_task import bench_watchdog_task
from quant_fund.models.wcet_est import bench_wcet_est


def test_edf_keys() -> None:
    assert "synthetic_edf_miss_free" in bench_edf_scheduler(1)


def test_rms_keys() -> None:
    out = bench_rms_scheduler(2)
    assert "synthetic_rms_ll_sound" in out


def test_wcet_keys() -> None:
    assert "synthetic_wcet_valid" in bench_wcet_est(3)


def test_debounce_keys() -> None:
    assert "synthetic_debounce_clean" in bench_debounce_fsm(4)


def test_watchdog_keys() -> None:
    assert "synthetic_watchdog_detects" in bench_watchdog_task(5)


def test_ring_keys() -> None:
    assert "synthetic_ring_order" in bench_ring_buffer(6)


def test_ranges() -> None:
    assert 0.0 <= bench_debounce_fsm(7)["synthetic_debounce_clean"] <= 1.0
    assert 0.0 <= bench_ring_buffer(8)["synthetic_ring_order"] <= 1.0


def test_determinism() -> None:
    assert bench_edf_scheduler(9) == bench_edf_scheduler(9)
