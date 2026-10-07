"""Tests for quant_fund.models.greek_neutral_portfolios — Taming the Greeks.

References: Tan, W.L., Roberts, S. & Zohren, S. (2026), "Taming the Greeks:
Option Portfolios with Inductive Biases", arXiv:2609.33767 (q-fin.PM), their
eqs. (1)–(15) and §5.3 penalty variants; Black, F. & Scholes, M. (1973) /
Merton, R.C. (1973) via quant_fund.models.options; Sharpe, W.F. (1994, the
performance-objective ratio form); Kingma, D.P. & Ba, J. (2014, Adam);
Hamilton, J.D. (1989, regime switching, composed via deep_hedging).

All data here is SYNTHETIC (seeded GBM / regime-switch path ensembles marked
under a flat BSM implied vol) — algorithmic-correctness evidence for the
framework, never market evidence; no live-trading claims. The Sharpe-like
training objective is a simulator-internal training signal reported only
under ``sim_internal_*`` keys, never a headline metric (AGENTS.md honesty
contract). Torch tests skip cleanly when the nn extra is absent.

The §6.3-style sweep configuration (module docstring of the source module
documents the mechanism): the training ensemble (GBM, positive drift) rewards
delta accumulation that rides the drift; the evaluation ensemble
(regime-switch, opposite drift) makes that directional tilt uncompensated, so
calibrated drift-penalty regularization improves the OOS risk-adjusted
objective while realized exposure falls monotonically — the paper's central
trade-off, asserted below with wide deterministic margins.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from quant_fund.models import greek_neutral_portfolios as gnp
from quant_fund.models.deep_hedging import simulate_gbm_paths, simulate_regime_switch_paths
from quant_fund.models.options import bs_greeks, bs_price
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="greek-neutral portfolio training requires the nn extra (torch)"
)

S0 = 100.0
SIGMA = 0.20  # flat BSM implied vol for construction and marking
DT = 1.0 / 252.0
STEPS = 24
MONEYNESSES = (0.92, 0.96, 1.00, 1.04, 1.08)
MATURITIES = (0.10, 0.20)
DP_ALPHAS = (0.0, 0.3, 1.0, 3.0, 25.0)
ENP_ALPHAS = (0.0, 0.3, 1.0, 2.0, 3.0)
EPOCHS = 300
LR = 0.05
SEED = 0
MONO_TOL = 1e-4  # float32 Adam ulp slack for weak-monotonicity assertions


def _train_paths(n: int = 128, seed: int = 11) -> np.ndarray:
    """SYNTHETIC training ensemble: GBM with positive drift (delta rewarded)."""
    return gnp.simulate_path_ensemble(n, STEPS, s0=S0, mu=0.6, sigma=0.20, dt=DT, seed=seed)


def _eval_paths(n: int = 384, seed: int = 99) -> np.ndarray:
    """SYNTHETIC evaluation ensemble: regime-switch with opposite drift."""
    return gnp.simulate_path_ensemble(
        n,
        STEPS,
        kind="regime_switch",
        s0=S0,
        mu=-0.3,
        sigma_low=0.25,
        sigma_high=0.60,
        p_stay_low=0.90,
        p_stay_high=0.90,
        initial_regime=0.0,
        dt=DT,
        seed=seed,
    )


@pytest.fixture(scope="module")
def universe() -> gnp.StraddleUniverse:
    return gnp.build_straddle_universe(
        [m * S0 for m in MONEYNESSES], list(MATURITIES), s0=S0, sigma=SIGMA, r=0.0
    )


@pytest.fixture(scope="module")
def sweep_marks(
    universe: gnp.StraddleUniverse,
) -> tuple[gnp.StraddleBookMarks, gnp.StraddleBookMarks]:
    """Seeded SYNTHETIC train/eval marks with independent ensembles."""
    return (
        gnp.mark_straddle_book(universe, _train_paths(), dt=DT),
        gnp.mark_straddle_book(universe, _eval_paths(), dt=DT),
    )


@pytest.fixture(scope="module")
def small_marks(universe: gnp.StraddleUniverse) -> gnp.StraddleBookMarks:
    return gnp.mark_straddle_book(universe, _train_paths(n=48), dt=DT)


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_simulate_path_ensemble_delegates_and_fails_closed() -> None:
    a = gnp.simulate_path_ensemble(8, 5, kind="gbm", seed=3, s0=S0, mu=0.1, sigma=0.2, dt=DT)
    b = simulate_gbm_paths(8, 5, seed=3, s0=S0, mu=0.1, sigma=0.2, dt=DT)
    assert np.array_equal(a, b)
    c = gnp.simulate_path_ensemble(
        8,
        5,
        kind="regime_switch",
        seed=4,
        s0=S0,
        sigma_low=0.1,
        sigma_high=0.4,
        p_stay_low=0.9,
        p_stay_high=0.9,
        initial_regime=0.0,
        dt=DT,
    )
    d = simulate_regime_switch_paths(
        8,
        5,
        seed=4,
        s0=S0,
        sigma_low=0.1,
        sigma_high=0.4,
        p_stay_low=0.9,
        p_stay_high=0.9,
        initial_regime=0,
        dt=DT,
    )
    assert np.array_equal(c, d)
    with pytest.raises(ValueError, match="unknown path ensemble kind"):
        gnp.simulate_path_ensemble(4, 3, kind="heston")
    with pytest.raises(ValueError, match="n_paths"):
        gnp.simulate_path_ensemble(0, 3)


def test_build_straddle_universe_is_delta_neutral_at_inception(
    universe: gnp.StraddleUniverse,
) -> None:
    """Eq. (4) weights: each straddle starts delta-neutral (composition over bs_greeks)."""
    assert universe.n_opts == len(MONEYNESSES) * len(MATURITIES)
    assert np.all(universe.call_weights > 0.0) and np.all(universe.put_weights > 0.0)
    assert np.allclose(universe.call_weights + universe.put_weights, 1.0)
    for j in range(universe.n_opts):
        gc = bs_greeks(S0, universe.strikes[j], universe.maturities[j], SIGMA, 0.0, call=True)
        gp = bs_greeks(S0, universe.strikes[j], universe.maturities[j], SIGMA, 0.0, call=False)
        # The paper's pre-normalization weights are -Delta_P and Delta_C.
        assert universe.call_weights[j] == pytest.approx(-gp["delta"], rel=1e-12)
        assert universe.put_weights[j] == pytest.approx(gc["delta"], rel=1e-12)
        inception_delta = (
            universe.call_weights[j] * gc["delta"] + universe.put_weights[j] * gp["delta"]
        )
        assert abs(inception_delta) < 1e-12


def test_build_straddle_universe_fails_closed() -> None:
    with pytest.raises(ValueError, match="strikes"):
        gnp.build_straddle_universe([], [0.1])
    with pytest.raises(ValueError, match="strikes"):
        gnp.build_straddle_universe([-92.0], [0.1])
    with pytest.raises(ValueError, match="maturities"):
        gnp.build_straddle_universe([92.0], [0.0])
    with pytest.raises(ValueError, match="sigma"):
        gnp.build_straddle_universe([92.0], [0.1], sigma=0.0)
    with pytest.raises(ValueError, match="s0"):
        gnp.build_straddle_universe([92.0], [0.1], s0=float("nan"))


def test_vectorized_bsm_matches_options_module() -> None:
    """The pathwise marking twin is pinned to options.bs_price / bs_greeks."""
    for s in (90.0, 100.0, 112.5):
        for k in (95.0, 105.0):
            for tau in (0.03, 0.25):
                for sig in (0.15, 0.30):
                    for r in (0.0, 0.03):
                        c, p, dc, dp, gv = gnp._bs_call_put(
                            np.array([[s]]), k, np.array([[tau]]), sig, r
                        )
                        gc = bs_greeks(s, k, tau, sig, r, call=True)
                        gp = bs_greeks(s, k, tau, sig, r, call=False)
                        assert c[0, 0] == pytest.approx(
                            bs_price(s, k, tau, sig, r, call=True), rel=1e-12
                        )
                        assert p[0, 0] == pytest.approx(
                            bs_price(s, k, tau, sig, r, call=False), rel=1e-12
                        )
                        assert dc[0, 0] == pytest.approx(gc["delta"], abs=1e-12)
                        assert dp[0, 0] == pytest.approx(gp["delta"], abs=1e-12)
                        assert gv[0][0, 0] == pytest.approx(gc["gamma"], rel=1e-12)
                        assert gv[0][0, 0] == pytest.approx(gp["gamma"], rel=1e-12)
                        assert gv[1][0, 0] == pytest.approx(gc["vega"], rel=1e-12)


def test_mark_straddle_book_matches_bs_composition(universe: gnp.StraddleUniverse) -> None:
    paths = np.array(
        [
            [100.0, 103.0, 97.0, 101.0, 99.0],
            [100.0, 96.0, 92.0, 95.0, 98.0],
        ]
    )
    marks = gnp.mark_straddle_book(universe, paths, dt=DT)
    n_paths, n_marks = paths.shape
    n_steps = n_marks - 1
    assert marks.values.shape == (universe.n_opts, n_paths, n_marks)
    assert marks.returns.shape == (universe.n_opts, n_paths, n_steps)
    assert marks.delta.shape == marks.gamma.shape == marks.vega.shape == marks.returns.shape
    assert np.all(marks.values > 0.0) and np.all(np.isfinite(marks.returns))
    assert np.all(marks.scales == 1.0)  # vol targeting off by default
    for j in range(universe.n_opts):
        k_j = float(universe.strikes[j])
        wc = float(universe.call_weights[j])
        wp = float(universe.put_weights[j])
        for p_idx in range(n_paths):
            for kk in range(n_marks):
                tau = float(universe.maturities[j]) - kk * DT
                s = paths[p_idx, kk]
                expected_v = wc * bs_price(s, k_j, tau, SIGMA, 0.0, True) + wp * bs_price(
                    s, k_j, tau, SIGMA, 0.0, False
                )
                assert marks.values[j, p_idx, kk] == pytest.approx(expected_v, rel=1e-12)
                if kk < n_steps:
                    gc = bs_greeks(s, k_j, tau, SIGMA, 0.0, call=True)
                    gp = bs_greeks(s, k_j, tau, SIGMA, 0.0, call=False)
                    assert marks.delta[j, p_idx, kk] == pytest.approx(
                        wc * gc["delta"] + wp * gp["delta"], abs=1e-12
                    )
                    assert marks.gamma[j, p_idx, kk] == pytest.approx(gc["gamma"], rel=1e-12)
                    assert marks.vega[j, p_idx, kk] == pytest.approx(gc["vega"], rel=1e-12)
        # Returns are mark-over-mark relative changes (their eq. (2)).
        expected_r = (marks.values[j, :, 1:] - marks.values[j, :, :-1]) / marks.values[j, :, :-1]
        assert np.allclose(marks.returns[j], expected_r)


def test_mark_straddle_book_expiry_intrinsic_branch() -> None:
    """Expired contracts mark at weighted intrinsic value with intrinsic delta."""
    exp_uni = gnp.build_straddle_universe([100.0], [0.01], s0=S0, sigma=SIGMA)
    paths = np.array(
        [[100.0, 102.0, 99.0, 105.0, 108.0, 110.0], [100.0, 98.0, 97.0, 94.0, 92.0, 90.0]]
    )
    marks = gnp.mark_straddle_book(exp_uni, paths, dt=DT)
    wc = float(exp_uni.call_weights[0])
    wp = float(exp_uni.put_weights[0])
    # tau = 0.01 - k/252: live at k=0,1,2; expired at k=3,4,5 (marks) / k=3,4 (decisions).
    assert marks.values[0, 0, 3] == pytest.approx(wc * 5.0, rel=1e-12)  # S=105 > K
    assert marks.values[0, 0, 4] == pytest.approx(wc * 8.0, rel=1e-12)
    assert marks.values[0, 0, 5] == pytest.approx(wc * 10.0, rel=1e-12)
    assert marks.values[0, 1, 3] == pytest.approx(wp * 6.0, rel=1e-12)  # S=94 < K
    assert marks.values[0, 1, 5] == pytest.approx(wp * 10.0, rel=1e-12)
    assert marks.delta[0, 0, 3] == pytest.approx(wc, abs=1e-12)
    assert marks.delta[0, 0, 4] == pytest.approx(wc, abs=1e-12)
    assert marks.delta[0, 1, 3] == pytest.approx(-wp, abs=1e-12)
    assert marks.gamma[0, 0, 3] == 0.0 and marks.vega[0, 1, 4] == 0.0


def test_mark_straddle_book_fails_closed(universe: gnp.StraddleUniverse) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        gnp.mark_straddle_book(universe, np.array([[100.0, -1.0, 101.0]]), dt=DT)
    with pytest.raises(ValueError, match="dt"):
        gnp.mark_straddle_book(universe, np.array([[100.0, 101.0]]), dt=0.0)
    with pytest.raises(ValueError, match="StraddleUniverse"):
        gnp.mark_straddle_book("not-a-universe", np.array([[100.0, 101.0]]), dt=DT)  # type: ignore[arg-type]
    # Degenerate expiry exactly at the money: zero intrinsic straddle mark.
    exp_uni = gnp.build_straddle_universe([100.0], [0.001], s0=S0, sigma=SIGMA)
    with pytest.raises(ValueError, match="strictly positive"):
        gnp.mark_straddle_book(exp_uni, np.full((2, 6), 100.0), dt=DT)


def test_vol_target_scales_are_no_lookahead_and_fail_closed(
    universe: gnp.StraddleUniverse,
) -> None:
    paths = _train_paths(n=6, seed=5)
    marks = gnp.mark_straddle_book(universe, paths, dt=DT, vol_target=0.15, ewma_span=5)
    assert np.all(marks.scales > 0.0) and np.all(np.isfinite(marks.scales))
    assert np.all(marks.scales[:, :, 0] == 1.0)  # no history at k=0
    decay = 2.0 / 6.0
    r = marks.returns
    v1 = r[:, :, 0] ** 2
    assert np.allclose(marks.scales[:, :, 1], 0.15 * math.sqrt(DT) / np.sqrt(v1))
    v2 = (1.0 - decay) * v1 + decay * r[:, :, 1] ** 2
    assert np.allclose(marks.scales[:, :, 2], 0.15 * math.sqrt(DT) / np.sqrt(v2))
    with pytest.raises(ValueError, match="vol_target"):
        gnp.mark_straddle_book(universe, paths, dt=DT, vol_target=0.0)
    with pytest.raises(ValueError, match="ewma_span"):
        gnp.mark_straddle_book(universe, paths, dt=DT, vol_target=0.15, ewma_span=0)
    # Degenerate (identically zero) returns -> zero EWMA variance: fail closed.
    with pytest.raises(ValueError, match="degenerate EWMA variance"):
        gnp._ewma_vol_scales(np.zeros((2, 3, 5)), dt=DT, vol_target=0.15, ewma_span=5)


def test_scaled_returns_and_portfolio_returns_hand_case(
    universe: gnp.StraddleUniverse,
) -> None:
    paths = _train_paths(n=4, seed=6)
    marks = gnp.mark_straddle_book(universe, paths, dt=DT)
    w = np.zeros(universe.n_opts)
    w[2] = 3.0
    scaled = gnp.scaled_straddle_returns(marks, w)
    assert np.allclose(scaled[2], 3.0 * marks.returns[2])
    assert np.allclose(scaled[[0, 1, 3]], 0.0)
    port = gnp.portfolio_returns(marks, w)
    assert port.shape == (4, STEPS)
    assert np.allclose(port, 3.0 * marks.returns[2] / universe.n_opts)  # eq. (1) 1/N mean
    with pytest.raises(ValueError, match="n_opts"):
        gnp.scaled_straddle_returns(marks, np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        gnp.scaled_straddle_returns(marks, np.full(universe.n_opts, np.nan))


def test_performance_objective_hand_values_and_fail_closed() -> None:
    """J = -sqrt(ppy) * mean/std: for x=[1,2,3,4], mean^2/var = 5 -> J = -sqrt(252*5)."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert gnp.performance_objective(x) == pytest.approx(-math.sqrt(252.0 * 5.0), rel=1e-12)
    assert gnp.performance_objective(x, periods_per_year=1.0) == pytest.approx(-math.sqrt(5.0))
    centered = np.array([0.01, -0.01, 0.02, -0.02])
    assert gnp.performance_objective(centered) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="at least two"):
        gnp.performance_objective(np.array([1.0]))
    with pytest.raises(ValueError, match="degenerate return variance"):
        gnp.performance_objective(np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        gnp.performance_objective(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="periods_per_year"):
        gnp.performance_objective(x, periods_per_year=0.0)


# Tiny hand-worked penalty case: w = [1, -2], exposure (2, 1, 2).
_W_HAND = np.array([1.0, -2.0])
_EXPO_HAND = np.array([[[0.1, 0.3]], [[0.5, -0.5]]])
_DENOM_HAND = np.array([[[0.02, 0.04]], [[0.01, 0.03]]])


def test_greek_neutrality_penalty_np_hand_values() -> None:
    """All five variants (their eqs. (9)–(13)) on a hand-computed example."""
    # contributions: [0.1, 0.3; -1.0, 1.0] -> sum|c| = 2.4, sum c^2 = 2.10
    assert gnp.greek_neutrality_penalty_np(
        _W_HAND, _EXPO_HAND, variant="naive_l1"
    ) == pytest.approx(2.4)
    # sum_Omega |X| = 2*1 + 2*2 = 6; sum_Omega X^2 = 2*1 + 2*4 = 10
    assert gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant="enp_l1") == pytest.approx(
        2.4 / 6.0, rel=1e-6
    )
    assert gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant="enp_l2") == pytest.approx(
        2.10 / 10.0, rel=1e-6
    )
    # sum |X*Gamma| = (0.02+0.04) + 2*(0.01+0.03) = 0.14; sum (X*Gamma)^2 = 0.002 + 4*0.001 = 0.006
    # (expected values carry the eps=1e-8 stabilizer of their eqs. (12)-(13))
    assert gnp.greek_neutrality_penalty_np(
        _W_HAND, _EXPO_HAND, variant="dp_l1", denominator=_DENOM_HAND
    ) == pytest.approx(2.4 / (0.14 + 1e-8), rel=1e-12)
    assert gnp.greek_neutrality_penalty_np(
        _W_HAND, _EXPO_HAND, variant="dp_l2", denominator=_DENOM_HAND
    ) == pytest.approx(2.10 / (0.006 + 1e-8), rel=1e-12)


def test_penalty_scale_invariance_vs_naive_shrinkage() -> None:
    """ENP/DP are invariant to X -> cX (their §5.3.1 fix); the naive penalty is not.

    The eps stabilizer breaks exact invariance at O(eps) relative level, so
    the assertion tolerance is 1e-5.
    """
    c = 3.7
    for variant in ("enp_l1", "enp_l2"):
        base = gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant=variant)
        scaled = gnp.greek_neutrality_penalty_np(c * _W_HAND, _EXPO_HAND, variant=variant)
        assert scaled == pytest.approx(base, rel=1e-5)
    for variant in ("dp_l1", "dp_l2"):
        base = gnp.greek_neutrality_penalty_np(
            _W_HAND, _EXPO_HAND, variant=variant, denominator=_DENOM_HAND
        )
        scaled = gnp.greek_neutrality_penalty_np(
            c * _W_HAND, _EXPO_HAND, variant=variant, denominator=_DENOM_HAND
        )
        assert scaled == pytest.approx(base, rel=1e-5)
    naive = gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant="naive_l1")
    naive_scaled = gnp.greek_neutrality_penalty_np(c * _W_HAND, _EXPO_HAND, variant="naive_l1")
    assert naive_scaled == pytest.approx(c * naive)  # shrinkage incentive: linear in scale


def test_greek_neutrality_penalty_np_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown penalty variant"):
        gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant="enp_l3")
    with pytest.raises(ValueError, match="denominator"):
        gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant="dp_l1")
    with pytest.raises(ValueError, match="denominator shape"):
        gnp.greek_neutrality_penalty_np(
            _W_HAND, _EXPO_HAND, variant="dp_l2", denominator=_DENOM_HAND[:, :, :1]
        )
    with pytest.raises(ValueError, match="1-D"):
        gnp.greek_neutrality_penalty_np(np.ones((2, 1)), _EXPO_HAND, variant="enp_l1")
    with pytest.raises(ValueError, match="ndim"):
        gnp.greek_neutrality_penalty_np(_W_HAND, np.array([0.1, 0.2]), variant="enp_l1")
    with pytest.raises(ValueError, match="n_opts"):
        gnp.greek_neutrality_penalty_np(np.ones(3), _EXPO_HAND, variant="enp_l1")
    with pytest.raises(ValueError, match="eps"):
        gnp.greek_neutrality_penalty_np(_W_HAND, _EXPO_HAND, variant="enp_l1", eps=0.0)
    with pytest.raises(ValueError, match="finite"):
        gnp.greek_neutrality_penalty_np(_W_HAND, np.full_like(_EXPO_HAND, np.inf), variant="enp_l1")


def test_evaluate_portfolio_exposure_hand_case() -> None:
    """Eqs. (14)–(15) with a single contract: net = mean delta, gross = mean |delta|."""
    uni = gnp.build_straddle_universe([100.0], [0.2], s0=S0, sigma=SIGMA)
    paths = np.array([[100.0, 110.0, 95.0, 102.0]])
    marks = gnp.mark_straddle_book(uni, paths, dt=DT)
    mean_delta = float(marks.delta[0].mean())
    mean_abs_delta = float(np.abs(marks.delta[0]).mean())
    ev = gnp.evaluate_portfolio(marks, [2.5])
    assert ev["net_delta_exposure_mean"] == pytest.approx(mean_delta, rel=1e-12)
    assert ev["gross_delta_exposure_mean"] == pytest.approx(mean_abs_delta, rel=1e-12)
    assert ev["gross_allocation"] == pytest.approx(2.5)
    ev_short = gnp.evaluate_portfolio(marks, [-1.0])
    assert ev_short["net_delta_exposure_mean"] == pytest.approx(-mean_delta, rel=1e-12)
    assert ev_short["gross_delta_exposure_mean"] == pytest.approx(mean_abs_delta, rel=1e-12)
    # Uniform rescaling leaves the normalized exposures invariant (their §6.2).
    ev_big = gnp.evaluate_portfolio(marks, [250.0])
    assert ev_big["net_delta_exposure_mean"] == pytest.approx(ev["net_delta_exposure_mean"])
    with pytest.raises(ValueError, match="positive gross allocation"):
        gnp.evaluate_portfolio(marks, [0.0])
    with pytest.raises(ValueError, match="unknown exposure greek"):
        gnp.evaluate_portfolio(marks, [1.0], variant="enp_l1", exposure_greek="theta")
    with pytest.raises(ValueError, match="unknown penalty variant"):
        gnp.evaluate_portfolio(marks, [1.0], variant="nope")


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module has no top-level torch import; torch entry points fail closed."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_gnp_no_torch", gnp.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_gnp_no_torch", probe)  # dataclasses resolves via sys.modules
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core stays usable while torch is blocked
    uni = probe.build_straddle_universe([92.0, 100.0], [0.1], s0=S0, sigma=SIGMA)
    paths = probe.simulate_path_ensemble(4, 3, seed=0)
    marks = probe.mark_straddle_book(uni, paths, dt=DT)
    assert marks.values.shape[0] == 2
    assert math.isfinite(
        probe.greek_neutrality_penalty_np([1.0, -1.0], marks.delta, variant="enp_l1")
    )
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.greek_neutrality_penalty(uni.strikes, marks.delta, variant="enp_l1")
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.train_greek_neutral_portfolio(marks, alpha=1.0, variant="enp_l1")


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@requires_torch
def test_torch_penalty_matches_numpy_twin() -> None:
    import torch

    rng = np.random.default_rng(0)
    w_np = rng.normal(size=4)
    expo_np = rng.normal(size=(4, 5, 7))
    den_np = np.abs(rng.normal(size=(4, 5, 7))) + 0.01
    w_t = torch.as_tensor(w_np, dtype=torch.float64)
    expo_t = torch.as_tensor(expo_np, dtype=torch.float64)
    den_t = torch.as_tensor(den_np, dtype=torch.float64)
    for variant in ("naive_l1", "enp_l1", "enp_l2"):
        got = gnp.greek_neutrality_penalty(w_t, expo_t, variant=variant)
        want = gnp.greek_neutrality_penalty_np(w_np, expo_np, variant=variant)
        assert float(got) == pytest.approx(want, rel=1e-10)
    for variant in ("dp_l1", "dp_l2"):
        got = gnp.greek_neutrality_penalty(w_t, expo_t, variant=variant, denominator=den_t)
        want = gnp.greek_neutrality_penalty_np(w_np, expo_np, variant=variant, denominator=den_np)
        assert float(got) == pytest.approx(want, rel=1e-10)


@requires_torch
def test_torch_penalty_gradient_flows() -> None:
    import torch

    w = torch.tensor([0.5, -0.3, 0.2], dtype=torch.float64, requires_grad=True)
    expo = torch.tensor([[[0.2, -0.1]], [[0.4, 0.3]], [[-0.5, 0.1]]], dtype=torch.float64)
    den = torch.full((3, 1, 2), 0.05, dtype=torch.float64)
    for variant in ("naive_l1", "enp_l1", "enp_l2"):
        pen = gnp.greek_neutrality_penalty(w, expo, variant=variant)
        grads = torch.autograd.grad(pen, w, retain_graph=True)[0]
        assert bool(torch.isfinite(grads).all()) and float(grads.abs().sum()) > 0.0
    pen_dp = gnp.greek_neutrality_penalty(w, expo, variant="dp_l2", denominator=den)
    grads_dp = torch.autograd.grad(pen_dp, w)[0]
    assert bool(torch.isfinite(grads_dp).all()) and float(grads_dp.abs().sum()) > 0.0


@requires_torch
def test_torch_penalty_fails_closed() -> None:
    import torch

    w = torch.tensor([0.5, -0.3])
    expo = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="torch tensor"):
        gnp.greek_neutrality_penalty(np.array([0.5, -0.3]), expo, variant="enp_l1")
    with pytest.raises(ValueError, match="1-D"):
        gnp.greek_neutrality_penalty(w.reshape(2, 1), expo, variant="enp_l1")
    with pytest.raises(ValueError, match="ndim"):
        gnp.greek_neutrality_penalty(w, torch.zeros(2), variant="enp_l1")
    with pytest.raises(ValueError, match="n_opts"):
        gnp.greek_neutrality_penalty(w, torch.zeros(3, 2, 2), variant="enp_l1")
    with pytest.raises(ValueError, match="denominator"):
        gnp.greek_neutrality_penalty(w, expo, variant="dp_l1")
    with pytest.raises(ValueError, match="denominator shape"):
        gnp.greek_neutrality_penalty(w, expo, variant="dp_l1", denominator=torch.zeros(2, 3, 3))
    with pytest.raises(ValueError, match="finite"):
        gnp.greek_neutrality_penalty(torch.tensor([0.5, float("nan")]), expo, variant="enp_l1")
    with pytest.raises(ValueError, match="unknown penalty variant"):
        gnp.greek_neutrality_penalty(w, expo, variant="l3")
    with pytest.raises(ValueError, match="eps"):
        gnp.greek_neutrality_penalty(w, expo, variant="enp_l1", eps=-1.0)


@requires_torch
def test_training_reduces_loss_bounded_weights_deterministic(
    small_marks: gnp.StraddleBookMarks,
) -> None:
    res = gnp.train_greek_neutral_portfolio(
        small_marks, alpha=1.0, variant="dp_l1", epochs=120, lr=LR, seed=7
    )
    assert res.weights.shape == (small_marks.n_opts,)
    assert np.all(np.abs(res.weights) <= 1.0)  # tanh admissibility bound
    assert np.all(np.isfinite(res.weights))
    for curve in (res.loss_curve, res.performance_curve, res.penalty_curve):
        assert curve.shape == (120,) and np.all(np.isfinite(curve))
    assert res.loss_curve[-1] < res.loss_curve[0]  # training reduced the total loss
    assert res.final_loss == pytest.approx(
        res.final_performance + 1.0 * res.final_penalty, rel=1e-6
    )
    again = gnp.train_greek_neutral_portfolio(
        small_marks, alpha=1.0, variant="dp_l1", epochs=120, lr=LR, seed=7
    )
    assert np.array_equal(res.weights, again.weights)  # seeded determinism
    other = gnp.train_greek_neutral_portfolio(
        small_marks, alpha=1.0, variant="dp_l1", epochs=120, lr=LR, seed=8
    )
    assert not np.array_equal(res.weights, other.weights)


@requires_torch
def test_training_fails_closed(small_marks: gnp.StraddleBookMarks) -> None:
    with pytest.raises(ValueError, match="alpha"):
        gnp.train_greek_neutral_portfolio(small_marks, alpha=-1.0, variant="enp_l1")
    with pytest.raises(ValueError, match="epochs"):
        gnp.train_greek_neutral_portfolio(small_marks, alpha=1.0, variant="enp_l1", epochs=0)
    with pytest.raises(ValueError, match="lr"):
        gnp.train_greek_neutral_portfolio(small_marks, alpha=1.0, variant="enp_l1", lr=0.0)
    with pytest.raises(ValueError, match="init_scale"):
        gnp.train_greek_neutral_portfolio(small_marks, alpha=1.0, variant="enp_l1", init_scale=0.0)
    with pytest.raises(ValueError, match="unknown penalty variant"):
        gnp.train_greek_neutral_portfolio(small_marks, alpha=1.0, variant="enp")
    with pytest.raises(ValueError, match="unknown exposure greek"):
        gnp.train_greek_neutral_portfolio(
            small_marks, alpha=1.0, variant="enp_l1", exposure_greek="theta"
        )
    with pytest.raises(ValueError, match="StraddleBookMarks"):
        gnp.train_greek_neutral_portfolio(small_marks.returns, alpha=1.0, variant="enp_l1")  # type: ignore[arg-type]


@requires_torch
def test_naive_penalty_signal_shrinkage_vs_scale_invariant_enp(
    small_marks: gnp.StraddleBookMarks,
) -> None:
    """Their §5.3.1: the naive L1 penalty collapses gross allocation; ENP does not."""
    naive = gnp.train_greek_neutral_portfolio(
        small_marks, alpha=1e-2, variant="naive_l1", epochs=200, lr=LR, seed=SEED
    )
    enp = gnp.train_greek_neutral_portfolio(
        small_marks, alpha=3.0, variant="enp_l1", epochs=200, lr=LR, seed=SEED
    )
    naive_alloc = float(np.abs(naive.weights).sum())
    enp_alloc = float(np.abs(enp.weights).sum())
    assert naive.penalty_curve[-1] < naive.penalty_curve[0]  # penalty was minimized
    assert naive_alloc < 0.25 * enp_alloc  # shrinkage is the naive penalty's trivial route


@requires_torch
def test_vega_exposure_penalty_trains(small_marks: gnp.StraddleBookMarks) -> None:
    """The framework is Greek-agnostic (their §5.3): vega exposure penalized like delta."""
    unreg = gnp.train_greek_neutral_portfolio(
        small_marks,
        alpha=0.0,
        variant="enp_l1",
        exposure_greek="vega",
        epochs=200,
        lr=LR,
        seed=SEED,
    )
    reg = gnp.train_greek_neutral_portfolio(
        small_marks,
        alpha=10.0,
        variant="enp_l1",
        exposure_greek="vega",
        epochs=200,
        lr=LR,
        seed=SEED,
    )

    def vega_pen(w: np.ndarray) -> float:
        return gnp.greek_neutrality_penalty_np(w, small_marks.vega, variant="enp_l1")

    assert reg.penalty_curve[-1] < reg.penalty_curve[0]
    assert vega_pen(reg.weights) < 0.5 * vega_pen(unreg.weights)  # vega exposure reduced


@requires_torch
def test_lambda_sweep_drift_penalty_exposure_monotone_and_interior_optimum(
    sweep_marks: tuple[gnp.StraddleBookMarks, gnp.StraddleBookMarks],
) -> None:
    """The paper's §6.3 central trade-off on the seeded SYNTHETIC ensembles.

    For the drift penalty (their DP-L1): realized directional exposure on the
    evaluation ensemble falls monotonically in alpha (their eqs. (14)–(15)
    diagnostics; net tilt driven toward zero), while the OOS risk-adjusted
    objective (negative annualized Sharpe of pooled scaled returns — a
    ``sim_internal`` training signal, never a headline metric) holds or
    improves up to a calibrated alpha and degrades once the penalty
    overwhelms the performance objective: an interior optimum.
    """
    train_marks, eval_marks = sweep_marks
    res = gnp.penalty_strength_sweep(
        train_marks,
        eval_marks,
        alphas=DP_ALPHAS,
        variant="dp_l1",
        epochs=EPOCHS,
        lr=LR,
        seed=SEED,
    )
    n = len(DP_ALPHAS)
    assert len(res.rows) == n
    assert np.array_equal(res.alphas, np.asarray(DP_ALPHAS))
    assert res.weights_by_alpha.shape == (n, train_marks.n_opts)
    gross = res.gross_delta_exposure_by_alpha
    abs_net = res.abs_net_delta_exposure_by_alpha
    perf = res.sim_internal_perf_objective_by_alpha

    # Leg 1 — monotone exposure: gross position-normalized delta (their eq.
    # (15), the stricter position-level measure) falls weakly monotonically.
    assert np.all(np.diff(gross) <= MONO_TOL)
    assert gross[-1] < 0.85 * gross[0]  # >=15% realized exposure reduction
    assert np.all(np.diff(abs_net) <= MONO_TOL)
    assert abs_net[-1] < 0.9 * abs_net[0]
    # The baseline's persistent net directional tilt is driven toward zero.
    net = res.net_delta_exposure_by_alpha
    assert abs(net[-1]) < 0.5 * abs(net[0])

    # Leg 2 — interior optimum of the OOS risk-adjusted objective.
    assert res.interior_optimum and 0 < res.best_index < n - 1
    assert perf[1] <= perf[0] + 1e-9  # holds-or-improves at first calibration
    gain = perf[0] - perf[res.best_index]
    assert gain > 0.03  # calibrated regularization beats the baseline (measured ~0.07)
    assert perf[-1] - perf[res.best_index] > 1.0  # extreme alpha degrades (measured ~3.2)

    # Metrics: honesty namespacing — any forbidden-token key must be
    # sim_internal-prefixed (mirrors the rl_market_maker honesty scan).
    for key, value in res.metrics.items():
        assert isinstance(value, float) and math.isfinite(value)
        tokens = key.lower().split("_")
        if any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in tokens):
            assert key.startswith("sim_internal_"), key
    assert res.metrics["gnp_interior_optimum"] == 1.0
    assert res.metrics["gnp_gross_delta_exposure_reduction"] > 0.0
    assert res.metrics["sim_internal_gnp_perf_objective_gain_best_vs_baseline"] == pytest.approx(
        gain, rel=1e-12
    )
    assert "sim_internal_gnp_perf_objective_oos_best" in res.metrics
    assert "gnp_gross_delta_exposure_baseline" in res.metrics
    assert "gnp_gross_delta_exposure_best" in res.metrics


@requires_torch
def test_lambda_sweep_exposure_normalized_penalty_exposure_monotone(
    sweep_marks: tuple[gnp.StraddleBookMarks, gnp.StraddleBookMarks],
) -> None:
    """Their ENP-L1: exposure falls monotonically without signal shrinkage.

    On this seeded ensemble the ENP variant buys exposure reduction at a
    monotone performance cost (no interior optimum — the paper likewise finds
    the trade-off shape is penalty-variant dependent, their §6.3); the scale
    invariance of eq. (10) keeps gross allocation healthy, in contrast to the
    naive penalty's shrinkage.
    """
    train_marks, eval_marks = sweep_marks
    res = gnp.penalty_strength_sweep(
        train_marks,
        eval_marks,
        alphas=ENP_ALPHAS,
        variant="enp_l1",
        epochs=EPOCHS,
        lr=LR,
        seed=SEED,
    )
    gross = res.gross_delta_exposure_by_alpha
    perf = res.sim_internal_perf_objective_by_alpha
    alloc = np.asarray([r["gross_allocation"] for r in res.rows])
    assert np.all(np.diff(gross) <= MONO_TOL) and gross[-1] < gross[0]
    assert np.all(np.diff(res.abs_net_delta_exposure_by_alpha) <= MONO_TOL)
    assert np.all(np.diff(perf) >= -1e-9)  # graceful, monotone performance cost
    assert alloc[-1] > 0.7 * alloc[0]  # no shrinkage collapse (scale-invariant penalty)


@requires_torch
def test_lambda_sweep_fails_closed(
    sweep_marks: tuple[gnp.StraddleBookMarks, gnp.StraddleBookMarks],
) -> None:
    train_marks, eval_marks = sweep_marks
    with pytest.raises(ValueError, match="at least three"):
        gnp.penalty_strength_sweep(train_marks, eval_marks, alphas=[0.0, 1.0], variant="enp_l1")
    with pytest.raises(ValueError, match="strictly increasing"):
        gnp.penalty_strength_sweep(
            train_marks, eval_marks, alphas=[0.0, 2.0, 1.0], variant="enp_l1"
        )
    with pytest.raises(ValueError, match="non-negative"):
        gnp.penalty_strength_sweep(
            train_marks, eval_marks, alphas=[-1.0, 0.0, 1.0], variant="enp_l1"
        )
    with pytest.raises(ValueError, match="unknown penalty variant"):
        gnp.penalty_strength_sweep(train_marks, eval_marks, alphas=[0.0, 1.0, 2.0], variant="x")
    uni_small = gnp.build_straddle_universe([95.0], [0.15], s0=S0, sigma=SIGMA)
    marks_small = gnp.mark_straddle_book(uni_small, _train_paths(n=4), dt=DT)
    with pytest.raises(ValueError, match="n_opts"):
        gnp.penalty_strength_sweep(
            marks_small, eval_marks, alphas=[0.0, 1.0, 2.0], variant="enp_l1"
        )


def test_dp_denominator_guard_message_is_not_inverted() -> None:
    """The dp_* invariant in greek_neutrality_penalty_np asserts that the
    denominator Greek must be SET — the pre-fix message raised the negation
    ('denominator is not None').  The guard is defensive/unreachable
    (_validate_penalty_inputs rejects dp_* with den=None first), so this
    pins the message text at source level."""
    import inspect

    src = inspect.getsource(gnp.greek_neutrality_penalty_np)
    assert "denominator must be set for the dp_* variants" in src
    assert 'raise ValueError("denominator is not None")' not in src
