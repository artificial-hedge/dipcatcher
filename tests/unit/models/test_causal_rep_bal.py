import pytest

from quant_fund.models.causal_rep_bal import bench_causal_rep_bal


def test_untrained_arms_identical() -> None:
    # with iters=0 neither arm trains, so the IPM penalty is inert — the
    # two CATEs must be EXACTLY equal iff the ablation arm shares the
    # penalized arm's initialization. The old code never re-seeded before
    # building phi2/h2, so the "ablation" was a different random net.
    pytest.importorskip("torch")
    out = bench_causal_rep_bal(iters=0)
    assert out["synthetic_crb_pehe"] == out["synthetic_crb_unbal_pehe"]


def test_bench_deterministic() -> None:
    pytest.importorskip("torch")
    a = bench_causal_rep_bal(iters=30)
    b = bench_causal_rep_bal(iters=30)
    assert a == b


def test_bench_returns_finite() -> None:
    pytest.importorskip("torch")
    out = bench_causal_rep_bal(iters=50)
    for _k, v in out.items():
        assert v == v  # no NaN
        assert v < float("inf")
