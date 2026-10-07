"""Adversarial probes for barrier_options."""

import pytest

from quant_fund.models import barrier_options as bo

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.03, 0.25


def test_knock_typo_rejected():
    with pytest.raises(ValueError, match="knock"):
        bo.down_call(S, K, S * 0.9, T, R, SIG, knock="knockout")
    with pytest.raises(ValueError, match="knock"):
        bo.up_put(S, K, S * 1.1, T, R, SIG, knock="IN")
    with pytest.raises(ValueError, match="knock"):
        bo.up_call(S, K, S * 1.1, T, R, SIG, knock="")
    with pytest.raises(ValueError, match="knock"):
        bo.down_put(S, K, S * 0.9, T, R, SIG, knock="OUT")


def test_parity_identities_all_eight():
    """in + out == vanilla for every one of the 8 barrier types."""
    h_dn, h_up = S * 0.85, S * 1.15
    assert bo.down_call(S, K, h_dn, T, R, SIG, knock="in") + bo.down_call(
        S, K, h_dn, T, R, SIG, knock="out"
    ) == pytest.approx(bo.vanilla_call(S, K, T, R, SIG), abs=1e-8)
    assert bo.up_call(S, K, h_up, T, R, SIG, knock="in") + bo.up_call(
        S, K, h_up, T, R, SIG, knock="out"
    ) == pytest.approx(bo.vanilla_call(S, K, T, R, SIG), abs=1e-8)
    assert bo.down_put(S, K, h_dn, T, R, SIG, knock="in") + bo.down_put(
        S, K, h_dn, T, R, SIG, knock="out"
    ) == pytest.approx(bo.vanilla_put(S, K, T, R, SIG), abs=1e-8)
    assert bo.up_put(S, K, h_up, T, R, SIG, knock="in") + bo.up_put(
        S, K, h_up, T, R, SIG, knock="out"
    ) == pytest.approx(bo.vanilla_put(S, K, T, R, SIG), abs=1e-8)


def test_out_price_within_vanilla():
    van = bo.vanilla_call(S, K, T, R, SIG)
    for h in (S * 0.7, S * 0.9, S * 0.98):
        p = bo.down_call(S, K, h, T, R, SIG, knock="out")
        assert -1e-9 <= p <= van + 1e-9


def test_bench_smoke():
    out = bo.bench_barrier_options()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_parity_max_rel_err"] < 1e-5
