from quant_fund.models.buffer_pool import bench_buffer_pool, clock_run, fifo_run


def test_flush_preserves_every_write():
    """Probe: the old check asserted disk[p] == buf[p] or disk[p] > 0 on
    resident pages — it failed when a never-evicted resident page had
    disk 0, and could not detect a lost write at all. The metric must
    be 1.0 exactly: every access lands on disk after final flush."""
    out = bench_buffer_pool()
    assert out["synthetic_flush_preserves_writes"] == 1.0


def test_hot_pages_hit_is_real_rate_not_trace_fraction():
    out = bench_buffer_pool()
    # real hit rate on hot accesses under min-id eviction — the old
    # check counted the trace's ~0.8 hot fraction (a generator property
    # the sim never touched) and always passed; a measured hit rate
    # under this policy is far below the trace skew
    assert 0.0 <= out["synthetic_hot_pages_hit"] < 0.6


def test_clock_beats_fifo_on_skewed_trace():
    trace = [0, 1, 2, 0, 1, 2, 9, 0, 1, 2, 10, 0, 1, 2] * 4
    assert clock_run(trace, 4) <= fifo_run(trace, 4)


def test_fifo_basic():
    assert fifo_run([1, 2, 3, 1], 2) == 4
    assert fifo_run([1, 1, 1], 1) == 1


def test_bench_runs():
    out = bench_buffer_pool()
    assert 0.0 < out["synthetic_clock_miss_ratio"] <= 1.0
