"""Probes for zkp_schnorr — sigma-protocol properties."""

import random

from quant_fund.models.zkp_schnorr import G, P, Q, bench_zkp_schnorr


def test_extraction_is_deterministic() -> None:
    """Two transcripts on one commitment extract w — check the algebra directly."""
    rng = random.Random(9)
    w = rng.randrange(1, Q)
    pub = pow(G, w, P)
    r = rng.randrange(1, Q)
    assert pow(G, r, P) > 0  # commitment t = G^r
    c1, c2 = 3, 7
    s1 = (r + c1 * w) % Q
    s2 = (r + c2 * w) % Q
    w_ext = (s1 - s2) * pow(c1 - c2, -1, Q) % Q
    assert pow(G, w_ext, P) == pub
    assert w_ext == w


def test_bench_all_pass() -> None:
    r = bench_zkp_schnorr()
    assert r["synthetic_completeness"] == 1.0
    assert r["synthetic_special_soundness"] == 1.0
    assert r["synthetic_simulator_verifies"] == 1.0
