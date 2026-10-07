import random

from quant_fund.models.bgw_mpc import bench_bgw_mpc, eval_circuit, reconstruct, share


def test_reconstruct_roundtrip():
    rng = random.Random(0)
    s = share(77, 3, 1, 101, rng)
    assert reconstruct(s, 101) == 77
    assert reconstruct(s[:2], 101) == 77


def test_lone_share_consistent_with_any_secret():
    # non-vacuous privacy: a single (x,y) share, x != 0, lies on a
    # degree-1 poly through (0,v) for EVERY candidate secret v.
    from quant_fund.models.bgw_mpc import _share_consistent

    rng = random.Random(1)
    p = 101
    s = share(42, 3, 1, p, rng)
    assert all(_share_consistent(s[0], v, p) for v in (0, 1, 42, 77, p - 1))


def test_resharing_randomizes_share_value():
    rng = random.Random(2)
    s1 = share(42, 3, 1, 101, rng)
    s2 = share(42, 3, 1, 101, rng)
    assert s1[0][1] != s2[0][1]


def test_eval_circuit_mixed_ops():
    rng = random.Random(3)
    out = eval_circuit(
        {"a": 3, "b": 4, "c": 5},
        [("*", "a", "b", "m"), ("+", "m", "c", "z")],
        3,
        1,
        101,
        rng,
    )
    assert out == 17


def test_bench_all_checks_pass():
    assert bench_bgw_mpc() == {"synthetic_bgw_mpc": 1.0}
