from quant_fund.models.negascout import (
    alpha_beta,
    bench_negascout,
    negascout,
    race_game,
)


def test_ns_matches_ab_values():
    children, terminal = race_game(14)
    for s in range(1, 15):
        v_ns, _ = negascout(s, children, terminal)
        v_ab, _ = alpha_beta(s, children, terminal)
        assert v_ns == v_ab


def test_bench():
    out = bench_negascout(seed=4)
    assert out["synthetic_values_match"] == 4.0
    assert 0.0 < out["synthetic_ns_node_ratio"] <= 1.0
