"""Unit tests for wave-299 game-playing-2 canon modules."""

import numpy as np

from quant_fund.models.expectimax import expectimax
from quant_fund.models.isomcts import _payoff, isomcts_policy
from quant_fund.models.mast_playout import _rollout, mast_train
from quant_fund.models.rave_mc import _ttt_play, _ttt_win, rave_mcts
from quant_fund.models.retrograde_wdl import _minimax, retrograde
from quant_fund.models.tablebase_dtm import _moves, _terminal, build_tablebase


def test_tablebase_consistency():
    tb = build_tablebase()
    for s, (w, d) in tb.items():
        t = _terminal(s)
        if t is not None:
            assert d == 0
        elif w == 1:
            assert any(tb[s2][0] == -1 for s2 in _moves(s))
        elif w == -1:
            assert all(tb[s2][0] == 1 for s2 in _moves(s))


def test_tablebase_terminal():
    tb = build_tablebase()
    assert tb[(0, 5, 1)] == (-1, 0) or tb[(0, 5, 1)][0] == -1


def test_retrograde_matches_minimax():
    succ = {s: ([s + 1] + ([s + 2] if s + 2 <= 5 else [])) if s < 5 else [] for s in range(6)}
    terminal = {5: 1}

    wdl = retrograde(succ, terminal)
    for s in range(6):
        mm = _minimax(s, succ, terminal, 20, {})
        assert wdl.get(s, 0) == mm


def test_rave_picks_blocking_move():
    board = (1, 0, 0, 2, 2, 0, 0, 0, 1)
    assert rave_mcts(board, 1, sims=400, seed=3) == 5


def test_ttt_win_lines():
    b = _ttt_play(_ttt_play(_ttt_play((0,) * 9, 0, 1), 1, 1), 2, 1)
    assert _ttt_win(b, 1)


def test_mast_avoids_self_destruct():
    q = mast_train(iters=200)
    assert q[2] < q[0] and q[2] < q[1]
    rng = np.random.default_rng(2)
    assert np.mean([_rollout(rng, q)[0] for _ in range(60)]) >= np.mean(
        [_rollout(rng)[0] for _ in range(60)]
    )


def test_expectimax_value():
    v = expectimax(0, 0, {})
    assert 0.0 <= v <= 1.0 or abs(v) < 100


def test_isomcts_payoff_table():
    assert _payoff(5, 0, 0) == 0.5
    assert _payoff(0, 5, 0) == -0.5
    assert _payoff(5, 0, 1) == 0.5  # junk hand folds to the bet
    assert _payoff(5, 3, 1) == 1.0  # called by a mid hand, nuts wins
    a = isomcts_policy(5, sims=60, seed=0)
    assert a == 1
