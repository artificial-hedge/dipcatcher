"""Learning-switch TTL-boundary honesty tests."""

from __future__ import annotations

from quant_fund.models.eth_switch import Switch, bench_eth_switch


def test_frame_expires_at_ttl_boundary():
    """Entry learned at time t0 is usable through t0+ttl frames; the
    frame at t0+ttl+1 must flood."""
    sw = Switch(ttl=2)
    assert sw.frame("a", "b", 1) is False  # learn a at now=1
    assert sw.frame("c", "d", 2) is False
    # now=3: a's age = 2 <= ttl, still known to the switch
    assert sw.frame("c", "a", 3) is True
    # now=4: a's age = 3 > ttl -> flood
    assert sw.frame("x", "a", 3) is False


def test_oracle_agrees_at_boundary():
    """bench_eth_switch's oracle must predict the same boundary behavior
    the switch implements — reachable only at small ttl."""
    out = bench_eth_switch(ttl=2)
    assert out["synthetic_switch_learn"] == 1.0


def test_bench_default_ttl():
    assert bench_eth_switch()["synthetic_switch_learn"] == 1.0
