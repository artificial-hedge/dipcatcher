import numpy as np

from quant_fund.models.mcts import bench_mcts, mcts, optimal_move


def test_optimal_move():
    assert optimal_move(31) == 3  # leaves 28 ≡ 0 mod 4
    assert optimal_move(8) == 1  # P-position: any take, returns 1


def test_mcts_visits_grow():
    rng = np.random.default_rng(0)
    visits, wins, playouts = mcts(11, 100, rng)
    assert playouts == 100
    assert visits[11] > 1


def test_bench_correct():
    out = bench_mcts(seed=2)
    assert out["synthetic_move_correct"] == 1.0
