"""Tests for models/implied_tree.py — Derman-Kani implied binomial tree."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.implied_tree import (
    ImpliedTree,
    arrow_debreu_prices,
    bench_implied_tree,
    implied_binomial_tree,
    synth_smile,
    tree_price,
    tree_price_path,
)


@pytest.fixture(scope="module")
def smile() -> dict[str, np.ndarray]:
    return synth_smile(seed=0)


@pytest.fixture(scope="module")
def tree(smile) -> ImpliedTree:
    return implied_binomial_tree(
        100.0,
        smile["strikes"],
        smile["calls"],
        r=0.02,
        q_carry=0.0,
        n_steps=12,
        tau=0.5,
        repair=True,
    )


class TestGuards:
    def test_bad_inputs(self, smile):
        ks, cs = smile["strikes"], smile["calls"]
        with pytest.raises(ValueError):
            implied_binomial_tree(-1.0, ks, cs, 0.02, 0.0, 12, 0.5)
        with pytest.raises(ValueError):
            implied_binomial_tree(100.0, ks[:2], cs[:2], 0.02, 0.0, 12, 0.5)
        with pytest.raises(ValueError):
            implied_binomial_tree(100.0, ks[::-1], cs[::-1], 0.02, 0.0, 12, 0.5)
        with pytest.raises(ValueError):
            implied_binomial_tree(100.0, ks, -cs, 0.02, 0.0, 12, 0.5)
        with pytest.raises(ValueError):
            implied_binomial_tree(100.0, ks, cs, 0.02, 0.0, 1, 0.5)
        with pytest.raises(ValueError):
            implied_binomial_tree(100.0, ks, cs, 0.02, 0.0, 12, -0.5)
        with pytest.raises(ValueError):
            synth_smile(spot=-1.0)

    def test_payoff_shape_guard(self, tree):
        with pytest.raises(ValueError):
            tree_price(tree, np.ones(4))
        with pytest.raises(ValueError):
            tree_price_path(tree, -1.0, np.ones(tree.nodes[-1].size))


class TestStructure:
    def test_level_shapes(self, tree):
        assert len(tree.nodes) == 13
        for n, lvl in enumerate(tree.nodes):
            assert lvl.size == n + 1
            assert np.all(np.diff(lvl) > 0)  # ascending node prices
        for p in tree.probs:
            assert ((p >= 0) & (p <= 1)).all()

    def test_ad_mass(self, tree):
        total = arrow_debreu_prices(tree).sum()
        assert total == pytest.approx(np.exp(-tree.r * tree.tau), rel=1e-6)

    def test_martingale(self, tree):
        leaf = tree.nodes[-1]
        ad = arrow_debreu_prices(tree)
        e_fwd = (ad @ leaf) / np.exp(-tree.r * tree.tau)
        assert e_fwd == pytest.approx(tree.spot * np.exp((tree.r - tree.q) * tree.tau), rel=1e-6)

    def test_zero_violations_on_benign_smile(self, tree):
        assert tree.violations == 0

    def test_deterministic(self, smile, tree):
        t2 = implied_binomial_tree(
            100.0,
            smile["strikes"],
            smile["calls"],
            0.02,
            0.0,
            12,
            0.5,
            repair=True,
        )
        np.testing.assert_array_equal(t2.nodes[-1], tree.nodes[-1])


class TestPricing:
    def test_reprices_smile(self, tree, smile):
        leaf = tree.nodes[-1]
        for k, c in zip(smile["strikes"], smile["calls"], strict=True):
            t = tree_price(tree, np.maximum(leaf - k, 0.0))
            assert abs(t - c) / max(c, 1.0) < 0.12

    def test_put_call_parity(self, tree, smile):
        leaf = tree.nodes[-1]
        k = float(smile["strikes"][4])
        call = tree_price(tree, np.maximum(leaf - k, 0.0))
        put = tree_price(tree, np.maximum(k - leaf, 0.0))
        df = np.exp(-tree.r * tree.tau)
        fwd = tree.spot * np.exp((tree.r - tree.q) * tree.tau)
        assert call - put == pytest.approx(df * (fwd - k), rel=0.05)

    def test_ko_bounded_and_consistent(self, tree, smile):
        leaf = tree.nodes[-1]
        k = float(smile["strikes"][4])
        van = tree_price(tree, np.maximum(leaf - k, 0.0))
        ko = tree_price_path(tree, 85.0, np.maximum(leaf - k, 0.0))
        deep = tree_price_path(tree, 1e-6, np.maximum(leaf - k, 0.0))
        assert 0.0 <= ko <= van
        assert deep == pytest.approx(van, rel=0.05)

    def test_deeper_barrier_knocks_more(self, tree, smile):
        leaf = tree.nodes[-1]
        k = float(smile["strikes"][4])
        pay = np.maximum(leaf - k, 0.0)
        low = tree_price_path(tree, 50.0, pay)
        high = tree_price_path(tree, 95.0, pay)
        assert low >= high


class TestFailClosed:
    def test_no_repair_raises_on_degenerate_input(self):
        # a nearly-flat call strip at tiny strikes is uncalibratable
        ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
        cs = np.array([30.0, 25.0, 20.0, 15.0, 30.0, 35.0])
        with pytest.raises(ValueError):
            implied_binomial_tree(100.0, ks, cs, 0.02, 0.0, 20, 0.5)


class TestBench:
    def test_keys_finite(self):
        blob = bench_implied_tree(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_no_forbidden_tokens(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_implied_tree(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_science(self):
        blob = bench_implied_tree(seed=0)
        assert blob["synthetic_smile_fit_err"] < 0.1
        assert blob["synthetic_density_mass_err"] < 1e-6
        assert blob["synthetic_martingale_err"] < 1e-6
        assert blob["synthetic_arb_violations"] == 0.0
        assert blob["synthetic_path_claim_bounds"] == 1.0
        assert blob["synthetic_ko_consistency"] == 1.0
        assert blob["synthetic_determinism"] == 1.0
