"""Tests for metrics/skill_ratings.py (wave 26)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.skill_ratings import (
    bench_skill_ratings,
    bradley_terry_loglik,
    bradley_terry_mle,
    elo_fit,
    elo_update,
    glicko2_fit,
    glicko2_update,
    predict_matrix,
    synth_games,
)


def _log(n_players=6, n_games=200, seed=0):
    return synth_games(n_players, n_games, seed=seed)


class TestElo:
    def test_winner_gains_loser_loses(self):
        r = np.full(4, 1500.0)
        r2 = elo_update(r, 0, 1, 1.0)
        assert r2[0] > 1500 and r2[1] < 1500
        assert r[0] == 1500  # input unchanged

    def test_sum_preserved(self):
        r = np.array([1600.0, 1400.0])
        r2 = elo_update(r, 0, 1, 1.0)
        assert abs(r2.sum() - r.sum()) < 1e-9

    def test_upset_moves_more(self):
        r = np.array([1800.0, 1200.0])
        win = elo_update(r, 1, 0, 1.0)  # underdog wins
        exp = elo_update(r, 0, 1, 1.0)  # favorite wins
        assert abs(win[1] - r[1]) > abs(exp[0] - r[0])

    def test_fit_recovers_order(self):
        outcomes, theta = _log(6, 300, seed=3)
        r = elo_fit(outcomes, 6, passes=3, seed=1)
        assert np.argsort(r)[-1] == np.argsort(theta)[-1]

    def test_invalid(self):
        with pytest.raises(ValueError):
            elo_update(np.ones(3), 0, 1, 2.0)
        with pytest.raises(ValueError):
            elo_update(np.ones(3), 0, 1, 1.0, k=-1)


class TestGlicko2:
    def test_rd_shrinks_over_games(self):
        # one game vs a tight (RD=30) opponent is informative -> RD drops
        r, rd, vol = glicko2_update(1500.0, 350.0, 0.06, 1500.0, 30.0, 1.0)
        assert 1300 < r < 1700 and rd < 350.0
        outcomes, _ = _log(4, 200, seed=9)
        st = glicko2_fit(outcomes, 4, seed=9)
        assert st.rd.mean() < 100.0

    def test_beat_stronger_opp_moves_rating_more(self):
        # beating a 2000-rated opponent is a bigger surprise than
        # beating a 1000-rated one -> larger rating gain
        r1, _, _ = glicko2_update(1500.0, 350.0, 0.06, 2000.0, 200.0, 1.0)
        r2, _, _ = glicko2_update(1500.0, 350.0, 0.06, 1000.0, 200.0, 1.0)
        assert r1 > r2 > 1500.0
        # tighter opponent rating deviation shrinks RD slightly more
        _, rd_tight, _ = glicko2_update(1500.0, 350.0, 0.06, 1500.0, 50.0, 1.0)
        _, rd_loose, _ = glicko2_update(1500.0, 350.0, 0.06, 1500.0, 350.0, 1.0)
        assert rd_tight <= rd_loose

    def test_fit_returns_state(self):
        outcomes, theta = _log(8, 300, seed=5)
        st = glicko2_fit(outcomes, 8, seed=2)
        assert st.rating.shape == st.rd.shape == st.vol.shape == (8,)
        assert (st.rd < 350).all()

    def test_invalid(self):
        with pytest.raises(ValueError):
            glicko2_update(1500, 0, 0.06, 1500, 350, 1.0)
        with pytest.raises(ValueError):
            glicko2_update(1500, 350, 0.06, 1500, 350, 1.5)


class TestBradleyTerry:
    def test_dominant_player_max(self):
        outcomes, theta = _log(6, 300, seed=7)
        w = bradley_terry_mle(outcomes, 6)
        assert np.argmax(w) == np.argmax(theta)

    def test_normalized(self):
        outcomes, _ = _log(4, 100, seed=0)
        w = bradley_terry_mle(outcomes, 4)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w > 0).all()

    def test_loglik_increases(self):
        outcomes, _ = _log(5, 150, seed=2)
        w = bradley_terry_mle(outcomes, 5)
        ll0 = bradley_terry_loglik(outcomes, np.full(5, 0.2))
        ll1 = bradley_terry_loglik(outcomes, w)
        assert ll1 > ll0

    def test_symmetric_matrix(self):
        w = np.array([0.5, 0.3, 0.2])
        p = predict_matrix(w)
        assert np.allclose(p + p.T, 1.0)
        assert p[0, 1] > p[1, 0]

    def test_invalid(self):
        outcomes, _ = _log(4, 50, seed=0)
        with pytest.raises(ValueError):
            bradley_terry_mle(outcomes, 1)
        with pytest.raises(ValueError):
            predict_matrix(np.array([0.0, 0.5]))


class TestSynth:
    def test_shapes(self):
        outcomes, theta = synth_games(8, 100, seed=0)
        assert outcomes.shape == (100, 3)
        assert theta.shape == (8,)
        assert set(np.unique(outcomes[:, 2])) <= {0, 1}

    def test_invalid(self):
        with pytest.raises(ValueError):
            synth_games(1, 10)
        with pytest.raises(ValueError):
            synth_games(4, 2)


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_skill_ratings(seed=21)
        assert blob
        assert all(np.isfinite(v) for v in blob.values())
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in blob:
            assert forbidden.isdisjoint(k.lower().split("_"))

    def test_bench_determinism(self):
        assert bench_skill_ratings(seed=23) == bench_skill_ratings(seed=23)
