from quant_fund.models.alpha_beta import (
    alpha_beta,
    alpha_beta_tt,
    bench_alpha_beta,
    exact_value,
    minimax,
    race_game,
)


def test_values_match_exact():
    children, terminal = race_game(15)
    for s in range(1, 16):
        v_mm, _ = minimax(s, children, terminal)
        v_ab, _ = alpha_beta(s, children, terminal)
        v_tt, _ = alpha_beta_tt(s, children, terminal)
        assert v_mm == v_ab == v_tt == exact_value(s)


def test_tt_cuts_nodes():
    children, terminal = race_game(21)
    _, n_mm = minimax(21, children, terminal)
    _, n_tt = alpha_beta_tt(21, children, terminal)
    assert n_tt < n_mm * 0.01


def test_bench():
    out = bench_alpha_beta(seed=1)
    assert out["synthetic_ab_correct"] == 1.0
    assert out["synthetic_tt_node_ratio"] < 0.01
