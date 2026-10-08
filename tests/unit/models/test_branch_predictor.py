from quant_fund.models.branch_predictor import bench_branch_predictor, bimodal, gshare


def test_bimodal_learns_always_taken():
    # counters init weakly-not-taken: each of the 64 slots misses its
    # first taken outcome, so accuracy converges to 1 with trace length
    assert bimodal([1] * 2000) > 0.95


def test_bimodal_learns_never_taken():
    assert bimodal([0] * 2000) > 0.95


def test_gshare_uses_history_on_alternating_blocks():
    # strongly correlated pattern: alternates in blocks of 3 — gshare's
    # history bits should match or beat the per-position bimodal table
    trace = [(1 if (i // 3) % 2 == 0 else 0) for i in range(480)]
    assert gshare(trace) >= bimodal(trace) - 0.05


def test_bench_is_deterministic_per_seed():
    a = bench_branch_predictor(seed=123)
    b = bench_branch_predictor(seed=123)
    assert a == b


def test_bench_rates_are_real_measurements():
    out = bench_branch_predictor()
    assert 0.0 <= out["synthetic_beats_always_taken"] <= 1.0
    assert 0.0 <= out["synthetic_gshare_ge_bimodal"] <= 1.0
    # biased-loop traces: bimodal should learn them almost every trial
    assert out["synthetic_beats_always_taken"] > 0.5
