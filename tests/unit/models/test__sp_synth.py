"""Probes for _sp_synth (tic-tac-toe + Kuhn poker fixtures)."""

import numpy as np
import pytest

from quant_fund.models._sp_synth import (
    encode_kuhn,
    encode_ttt,
    kuhn_exploit,
    kuhn_infosets,
    kuhn_random_strat,
    kuhn_terminal,
    kuhn_util,
    minimax_val,
    optimal_move,
    play_ttt,
    ttt_eval,
)


def _first_legal(e, legal):
    return legal[0]


def _oracle_self(e, legal):
    s = tuple(int(x) for x in e)
    return optimal_move(s, 1)


def test_minimax_initial_state_is_draw():
    assert minimax_val((0,) * 9, 1) == 0.0  # ttt is a draw with perfect play


def test_optimal_move_blocks_win():
    # X to move; O threatens row (3,4,5) via cells 3,4 → must block 5... or win
    s = (1, 1, 0, -1, -1, 0, 0, 0, 0)  # X can win at 2
    assert optimal_move(s, 1) == 2
    s2 = (1, 0, 0, -1, -1, 0, 1, 0, 0)  # X==O count, X to move, must block 5
    assert optimal_move(s2, 1) == 5


def test_optimal_move_rejects_terminal():
    s = (1, 1, 1, 0, 0, 0, 0, 0, 0)  # X already won
    with pytest.raises(ValueError):
        optimal_move(s, -1)
    full = (1, -1, 1, 1, -1, -1, -1, 1, 1)
    with pytest.raises(ValueError):
        optimal_move(full, 1)


def test_play_ttt_draws_between_oracles():
    assert play_ttt(_oracle_self, _oracle_self) == 0


def test_play_ttt_rejects_illegal_move():
    # dishonest policy: always plays cell 0 → overwrites occupied cell mid-game
    def cheat(e, legal):
        return 0

    with pytest.raises(ValueError):
        play_ttt(cheat, _oracle_self)


def test_play_ttt_rejects_out_of_range_move():
    def bad(e, legal):
        return 99

    with pytest.raises(ValueError):
        play_ttt(bad, _oracle_self)


def test_ttt_eval_oracle_scores():
    s_or, s_rd = ttt_eval(_oracle_self, games=10)
    assert s_or == 1.0  # oracle never loses to itself
    assert 0.0 <= s_rd <= 1.0


def test_kuhn_util_and_terminal():
    assert kuhn_util("bb", 2, 1) == 2.0  # K beats Q at showdown after bets
    assert kuhn_util("bp", 2, 1) == 1.0  # bet then pass: bettor wins
    assert kuhn_terminal("pbp")
    assert not kuhn_terminal("pb")
    with pytest.raises(ValueError):
        kuhn_util("pb", 0, 1)  # non-terminal eval is an error, not 0


def test_kuhn_infosets_reachable():
    infos = kuhn_infosets()
    assert len(infos) == 12  # 2 players x 3 cards x 2 valid hists each
    for player, card, h in infos:
        assert card in (0, 1, 2)
        assert len(h) % 2 == player
        assert not kuhn_terminal(h)


def test_kuhn_random_strat_exploitability_positive():
    v = kuhn_exploit(kuhn_random_strat())
    assert np.isfinite(v)
    assert v > 0  # random strat is exploitable in Kuhn poker


def test_kuhn_exploit_rejects_bad_strategy():
    def nan_strat(p, c, h):
        return np.array([np.nan, 0.5])

    def skewed_strat(p, c, h):
        return np.array([0.9, 0.9])  # does not sum to 1

    def negative_strat(p, c, h):
        return np.array([-0.5, 1.5])

    def wrong_len(p, c, h):
        return np.array([1.0])

    for fn in (nan_strat, skewed_strat, negative_strat, wrong_len):
        with pytest.raises(ValueError):
            kuhn_exploit(fn)


def test_encode_shapes():
    assert encode_ttt((1, -1, 0, 0, 0, 0, 0, 0, 0), 1).shape == (9,)
    assert encode_kuhn(0, 2, "pb").shape == (8,)
    assert encode_kuhn(1, 0, "").shape == (8,)
