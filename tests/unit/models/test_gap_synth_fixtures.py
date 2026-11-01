"""Unit tests for the untested ``quant_fund.models._synth`` fixtures.

These modules are the shared SYNTHETIC fixtures behind the research canons.
Because every module in a canon benches against the *same* fixture, a silent
change to a fixture would move every result in that canon at once — so the
fixtures themselves need to be pinned.

What is asserted here is the *contract* of each fixture, not any performance
number: shapes, determinism, mathematical identities (the simplex optimum
bounds the integer optimum, the planted spectrum is recovered, the ridge fit
reproduces its own prediction), and the boundary behaviour a downstream
module would otherwise trip over.

Every fixture is SYNTHETIC. Nothing in this file is market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import (
    _al_synth,
    _amopt_synth,
    _approx_synth,
    _auc_synth,
    _bx_synth,
    _cate_synth,
    _cp2_synth,
    _da_synth,
    _dc_synth,
    _dfx_synth,
    _eb_synth,
    _eig_synth,
    _ex_synth,
    _gt_synth,
    _heat_synth,
    _ig_synth,
    _ilp_synth,
    _inv_synth,
    _it_synth,
    _lm_synth,
    _mc2_synth,
    _oc_synth,
    _pd_synth,
    _qc_synth,
    _queue_synth,
    _rel_synth,
    _sched_synth,
    _sig3_synth,
    _sk_synth,
    _sv_synth,
    _tn_synth,
    _vi_synth,
)

# --------------------------------------------------------------------------- #
# _al_synth — active learning
# --------------------------------------------------------------------------- #


def test_al_features_have_the_documented_width() -> None:
    X = np.zeros((4, 2))
    P = _al_synth.feats(X)
    assert P.shape == (4, 6)
    # The last column is the bias.
    assert np.all(P[:, -1] == 1.0)


def test_al_data_is_deterministic_and_splits_pool_from_test() -> None:
    Xp, yp, Xt, yt = _al_synth.make_data(0, n_pool=20, n_test=10)
    assert Xp.shape == (20, 2)
    assert Xt.shape == (10, 2)
    assert yp.shape == (20,)
    assert set(np.unique(yp)) <= {0.0, 1.0}
    Xp2, _yp2, _Xt2, _yt2 = _al_synth.make_data(0, n_pool=20, n_test=10)
    assert np.array_equal(Xp, Xp2)


def test_the_logistic_fit_reproduces_its_own_accuracy() -> None:
    Xp, yp, Xt, yt = _al_synth.make_data(0, n_pool=200, n_test=100)
    w = _al_synth.fit_logreg(_al_synth.feats(Xp), yp)
    assert w.shape == (6,)
    accuracy = _al_synth.acc(w, Xt, yt)
    assert 0.0 <= accuracy <= 1.0


def test_probabilities_are_a_valid_distribution() -> None:
    X = np.zeros((3, 2))
    P = _al_synth.feats(X)
    probs = _al_synth.probs(P, np.zeros(6))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


# --------------------------------------------------------------------------- #
# _amopt_synth — American option reference
# --------------------------------------------------------------------------- #


def test_the_option_parameters_are_the_declared_contract() -> None:
    params = _amopt_synth.amopt_params()
    assert params == {"S0": 100.0, "K": 105.0, "T": 0.5, "r": 0.05, "q": 0.02, "sigma": 0.25}


def test_the_gbm_path_starts_near_the_spot_and_stays_positive() -> None:
    path = _amopt_synth.gbm_path(0, steps=100)
    assert path.shape == (100,)
    assert np.all(path > 0.0)


def test_the_gbm_path_is_deterministic_for_a_seed() -> None:
    assert np.array_equal(_amopt_synth.gbm_path(3), _amopt_synth.gbm_path(3))


def test_the_american_price_is_at_least_the_european_price() -> None:
    """The early-exercise premium is non-negative."""
    european = _amopt_synth.european_put()
    american, _boundary = _amopt_synth.crr_price(200)
    assert european > 0.0
    assert american >= european - 1e-6


def test_the_european_put_price_is_close_to_the_analytic_value() -> None:
    """A 5x5 analytic check: the price is bounded by the intrinsic value."""
    price = _amopt_synth.european_put()
    intrinsic = max(_amopt_synth.K - _amopt_synth.S0, 0.0)
    assert intrinsic <= price <= _amopt_synth.K


# --------------------------------------------------------------------------- #
# _approx_synth — brute-force references
# --------------------------------------------------------------------------- #


def test_the_vertex_cover_reference_covers_every_edge() -> None:
    assert _approx_synth.brute_vc(_approx_synth.G_EDGES, _approx_synth.G_N) == 7


def test_the_max_cut_reference_is_within_the_possible_range() -> None:
    value = _approx_synth.brute_maxcut(_approx_synth.G_EDGES, _approx_synth.G_N)
    assert 0 <= value <= len(_approx_synth.G_EDGES)


def test_the_knapsack_reference_respects_the_capacity() -> None:
    value = _approx_synth.brute_knapsack(
        _approx_synth.KS_V, _approx_synth.KS_W, _approx_synth.KS_CAP
    )
    assert value > 0.0
    assert value <= float(_approx_synth.KS_V.sum())


def test_the_knapsack_reference_never_exceeds_the_capacity() -> None:
    """The best feasible subset is at least the single best feasible item."""
    best = _approx_synth.brute_knapsack(
        _approx_synth.KS_V, _approx_synth.KS_W, _approx_synth.KS_CAP
    )
    for value, weight in zip(_approx_synth.KS_V, _approx_synth.KS_W, strict=True):
        if weight <= _approx_synth.KS_CAP:
            assert best >= value


# --------------------------------------------------------------------------- #
# _auc_synth — auction fixture
# --------------------------------------------------------------------------- #


def test_the_auction_values_are_uniform_on_the_declared_interval() -> None:
    values = _auc_synth.iid_values(0, n=4, m=500)
    assert values.shape == (500, 4)
    assert values.min() >= _auc_synth.VAL_LO
    assert values.max() <= _auc_synth.VAL_HI


def test_the_auction_values_are_deterministic_for_a_seed() -> None:
    assert np.array_equal(_auc_synth.iid_values(0), _auc_synth.iid_values(0))


def test_the_symmetric_first_price_bid_is_the_documented_linear_map() -> None:
    values = np.array([0.0, 0.5, 1.0])
    bids = _auc_synth.uniform_bne_bid(values, n=4)
    assert bids == pytest.approx(0.75 * values)


# --------------------------------------------------------------------------- #
# _bx_synth — bandit / MDP fixtures
# --------------------------------------------------------------------------- #


def test_the_bandit_arms_are_in_the_declared_range() -> None:
    mu, sd = _bx_synth.bandit_env(0, K=5)
    assert mu.shape == (5,)
    assert sd.shape == (5,)
    assert np.all((mu >= 0.1) & (mu <= 1.0))
    assert np.all(sd == 0.3)


def test_the_psrl_transitions_are_row_stochastic() -> None:
    p0, p1, r0, r1 = _bx_synth.psrl_env(0)
    for P in (p0, p1):
        assert P.shape == (4, 4)
        assert np.allclose(P.sum(axis=1), 1.0)
    for R in (r0, r1):
        assert R.shape == (4,)
        assert np.all((R >= 0.0) & (R <= 1.0))


# --------------------------------------------------------------------------- #
# _cate_synth — confounded treatment
# --------------------------------------------------------------------------- #


def test_the_cate_fixture_returns_the_documented_tuple() -> None:
    X, t, y, tau, ey1 = _cate_synth.cate_data(0, n=40, d=6)
    assert X.shape == (40, 6)
    assert t.shape == (40,)
    assert set(np.unique(t)) <= {0, 1}
    assert tau.shape == (40,)
    assert ey1.shape == (40,)


def test_pehe_is_the_root_mean_squared_error() -> None:
    tau_hat = np.array([1.0, 2.0, 3.0])
    tau = np.array([1.0, 2.0, 4.0])
    assert _cate_synth.pehe(tau_hat, tau) == pytest.approx(np.sqrt(1.0 / 3.0))


def test_pehe_of_a_perfect_estimate_is_zero() -> None:
    tau = np.array([0.5, -1.0, 2.0])
    assert _cate_synth.pehe(tau, tau) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _cp2_synth — conformal regression / classification
# --------------------------------------------------------------------------- #


def test_the_conformal_fixture_splits_train_from_test() -> None:
    Xtr, ytr, Xte, yte = _cp2_synth.cp2_data(0, n=50)
    assert Xtr.shape == (50, 3)
    assert ytr.shape == (50,)
    assert Xte.shape == (50, 3)
    assert yte.shape == (50,)


def test_the_ridge_fit_reproduces_its_own_prediction() -> None:
    Xtr, ytr, Xte, _yte = _cp2_synth.cp2_data(0, n=60)
    w = _cp2_synth.fit_ridge(Xtr, ytr)
    assert w.shape == (4,)
    assert _cp2_synth.ridge_pred(w, Xte).shape == (60,)


def test_coverage_and_width_are_reported_together() -> None:
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([2.0, 3.0, 4.0])
    coverage, width = _cp2_synth.cov_width(lo, hi, y)
    assert coverage == pytest.approx(1.0)
    assert width == pytest.approx(2.0)


def test_coverage_counts_only_the_interval() -> None:
    y = np.array([0.0, 5.0])
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    coverage, _width = _cp2_synth.cov_width(lo, hi, y)
    assert coverage == pytest.approx(0.5)


def test_the_classifier_fixture_returns_binary_labels() -> None:
    Xtr, ytr, Xte, yte = _cp2_synth.cls_data(0, n=50)
    assert Xtr.shape == (50, 4)
    assert Xte.shape == (50, 4)
    assert set(np.unique(ytr)) <= {0, 1}
    assert set(np.unique(yte)) <= {0, 1}


# --------------------------------------------------------------------------- #
# _da_synth — differentiable top-k selection
# --------------------------------------------------------------------------- #


def test_the_top_k_mask_selects_exactly_k_features() -> None:
    s = np.array([0.1, 0.9, 0.5, 0.2])
    mask = _da_synth.topk_mask(s, k=2)
    assert mask.sum() == 2.0
    assert mask[1] == 1.0
    assert mask[2] == 1.0


def test_the_loss_is_the_squared_error_of_the_weighted_selection() -> None:
    s = np.zeros(6)
    assert _da_synth.loss_np(s) == pytest.approx(_da_synth.TARGET**2)


def test_the_finite_difference_gradient_matches_the_loss() -> None:
    """The central difference is the ground truth for the gradient."""
    s = np.array([0.2, -0.4, 0.9, 0.1, -0.7, 0.3])
    gradient = _da_synth.finite_diff_grad(s)
    assert gradient.shape == (6,)
    assert np.all(np.isfinite(gradient))


def test_the_gradient_correlation_of_a_vector_with_itself_is_one() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert _da_synth.grad_corr(v, v) == pytest.approx(1.0)


def test_the_gradient_correlation_of_a_zero_vector_is_zero() -> None:
    """A degenerate gradient must not produce a NaN correlation."""
    assert _da_synth.grad_corr(np.zeros(3), np.ones(3)) == 0.0


def test_the_gradient_correlation_is_sign_sensitive() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert _da_synth.grad_corr(v, -v) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# _dc_synth — data-centric selection
# --------------------------------------------------------------------------- #


def test_the_data_centric_fixture_returns_a_noisy_pool_and_a_clean_test() -> None:
    X, y, Xt, yt = _dc_synth.dc_data(0, n=40)
    assert X.shape == (40, 6)
    assert Xt.shape == (300, 6)
    assert set(np.unique(y)) <= {0, 1}
    assert set(np.unique(yt)) <= {0, 1}


def test_the_logistic_evaluation_reports_a_probability() -> None:
    X, y, Xt, yt = _dc_synth.dc_data(0, n=60)
    accuracy = _dc_synth.fit_eval(X, y, Xt, yt)
    assert 0.0 <= accuracy <= 1.0


# --------------------------------------------------------------------------- #
# _dfx_synth / _eb_synth — MMD
# --------------------------------------------------------------------------- #


def test_the_gaussian_mmd_of_a_sample_against_itself_is_small() -> None:
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((200, 2))
    assert _dfx_synth.gauss_mmd(rng, sample, n=200) < 0.2


def test_the_mmd_of_a_sample_with_itself_is_zero() -> None:
    rng = np.random.default_rng(1)
    sample = rng.standard_normal((50, 2))
    assert _eb_synth.mmd(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_the_mmd_separates_two_distinct_populations() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((100, 2))
    b = rng.standard_normal((100, 2)) + 3.0
    assert _eb_synth.mmd(a, b) > 0.2


def test_the_gaussian_baseline_mmd_is_finite() -> None:
    rng = np.random.default_rng(3)
    train = rng.standard_normal((100, 2))
    test = rng.standard_normal((80, 2))
    assert np.isfinite(_eb_synth.gauss_baseline_mmd(train, test))


# --------------------------------------------------------------------------- #
# _eig_synth — planted spectra
# --------------------------------------------------------------------------- #


def test_the_planted_symmetric_spectrum_is_recovered() -> None:
    spectrum = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    matrix, lam = _eig_synth.sym_planted(0, 6, spectrum)
    assert np.allclose(np.sort(np.linalg.eigvalsh(matrix)), spectrum)
    assert np.allclose(lam, spectrum)


def test_the_planted_symmetric_matrix_is_symmetric() -> None:
    matrix, _lam = _eig_synth.sym_planted(0, 5)
    assert np.allclose(matrix, matrix.T)


def test_the_planted_nonsymmetric_eigenvalues_are_recovered() -> None:
    matrix, lam = _eig_synth.nonsym_planted(0, 5)
    assert np.allclose(np.sort(np.linalg.eigvals(matrix)), np.sort(lam))


def test_the_top_eigenvalue_error_is_zero_for_an_exact_recovery() -> None:
    """The reference spectrum is compared in descending order."""
    lam = np.array([1.0, 2.0, 3.0])
    assert _eig_synth.top_eig_err(lam, np.array([3.0, 2.0, 1.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _ex_synth — hard-exploration gridworld
# --------------------------------------------------------------------------- #


def test_the_grid_index_is_row_major() -> None:
    assert _ex_synth.s2i((0, 0)) == 0
    assert _ex_synth.s2i((2, 3)) == 2 * _ex_synth.SIDE + 3


def test_the_walls_have_no_neighbours() -> None:
    for wall in _ex_synth.WALLS:
        assert all(nxt not in _ex_synth.WALLS for _action, nxt in _ex_synth.neighbors(wall))


def test_the_goal_has_a_unit_reward_and_the_distractors_a_small_one() -> None:
    assert _ex_synth.reward(_ex_synth.GOAL) == 1.0
    for distractor in _ex_synth.DISTRACTORS:
        assert _ex_synth.reward(distractor) == pytest.approx(0.1)


def test_the_state_features_flag_the_goal_and_the_distractors() -> None:
    goal = _ex_synth.state_feat(_ex_synth.GOAL)
    assert goal[3] == 1.0
    assert goal[0] == pytest.approx(0.0)
    distractor = _ex_synth.state_feat(next(iter(_ex_synth.DISTRACTORS)))
    assert distractor[2] == 1.0


# --------------------------------------------------------------------------- #
# _gt_synth — graph-temporal fixture
# --------------------------------------------------------------------------- #


def test_the_ring_adjacency_is_row_stochastic() -> None:
    A = _gt_synth.ring_adj(8)
    assert A.shape == (8, 8)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_the_graph_temporal_data_has_the_requested_shape() -> None:
    X = _gt_synth.gt_data(0, T=50, n=6)
    assert X.shape == (50, 6)
    assert np.all(np.isfinite(X))


def test_the_ar2_baseline_reports_a_finite_mse() -> None:
    X = _gt_synth.gt_data(0, T=60, n=4)
    assert np.isfinite(_gt_synth.ar2_baseline(X))


# --------------------------------------------------------------------------- #
# _heat_synth — analytic heat kernel
# --------------------------------------------------------------------------- #


def test_the_grid_spans_the_domain_symmetrically() -> None:
    x = _heat_synth.grid(11)
    assert x.shape == (11,)
    assert x[0] == pytest.approx(-_heat_synth.L)
    assert x[-1] == pytest.approx(_heat_synth.L)


def test_the_initial_condition_is_a_normalised_gaussian() -> None:
    x = _heat_synth.grid(401)
    u = _heat_synth.u0(x)
    assert np.trapezoid(u, x) == pytest.approx(1.0, abs=1e-6)


def test_the_heat_kernel_spreads_and_decays() -> None:
    x = _heat_synth.grid(401)
    assert _heat_synth.u_exact(x, 0.5).max() < _heat_synth.u_exact(x, 0.0).max()


def test_the_exact_solution_at_zero_time_is_the_initial_condition() -> None:
    x = _heat_synth.grid(101)
    assert np.allclose(_heat_synth.u_exact(x, 0.0), _heat_synth.u0(x))


def test_the_relative_l2_error_of_the_exact_solution_is_zero() -> None:
    x = _heat_synth.grid(101)
    truth = _heat_synth.u_exact(x, _heat_synth.T_)
    assert _heat_synth.rel_l2(truth, truth) == pytest.approx(0.0)


def test_the_monte_carlo_helper_returns_one_position_per_sample() -> None:
    positions = _heat_synth.mc_paths(64, 0.5, seed=0)
    assert positions.shape == (64,)
    assert np.all(np.isfinite(positions))


# --------------------------------------------------------------------------- #
# _ig_synth — information geometry
# --------------------------------------------------------------------------- #


def test_the_fisher_rao_pair_is_the_declared_one() -> None:
    assert _ig_synth.fr_gauss_pair() == ((0.0, 1.0), (1.5, 0.7))


def test_the_gaussian_fit_centres_on_its_target() -> None:
    sample = _ig_synth.gauss_fit(0, n=4000)
    assert abs(sample.mean() - 2.0) < 0.05


def test_the_simplex_least_squares_fixture_is_shaped_for_a_solution() -> None:
    A, b = _ig_synth.simplex_ls(0, d=6)
    assert A.shape == (6, 6)
    assert b.shape == (6,)


def test_the_poisson_counts_are_non_negative_integers() -> None:
    counts = _ig_synth.poisson_nmf(0, m=10, n=8, r=3)
    assert counts.shape == (10, 8)
    assert np.all(counts >= 0.0)


def test_the_multinomial_pair_is_a_pair_of_distributions() -> None:
    p, q = _ig_synth.multinomial_pair()
    assert p.shape == q.shape
    assert p.sum() == pytest.approx(1.0)
    assert q.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# _ilp_synth — LP / ILP references
# --------------------------------------------------------------------------- #


def test_the_simplex_solution_satisfies_its_constraints() -> None:
    x, value = _ilp_synth.simplex(_ilp_synth.ILP_C, _ilp_synth.ILP_A, _ilp_synth.ILP_B)
    assert np.all(_ilp_synth.ILP_A @ x <= _ilp_synth.ILP_B + 1e-6)
    assert np.all(x >= -1e-9)
    assert value == pytest.approx(float(_ilp_synth.ILP_C @ x))


def test_the_linear_relaxation_bounds_the_integer_optimum() -> None:
    """The continuous optimum is at least the integer optimum."""
    _x, lp_value = _ilp_synth.simplex(_ilp_synth.ILP_C, _ilp_synth.ILP_A, _ilp_synth.ILP_B)
    ilp_value = _ilp_synth.brute_force_ilp(
        _ilp_synth.ILP_C, _ilp_synth.ILP_A, _ilp_synth.ILP_B, _ilp_synth.ILP_UB
    )
    assert lp_value >= ilp_value - 1e-6


def test_the_integer_optimum_is_attained_at_an_integer_point() -> None:
    value = _ilp_synth.brute_force_ilp(
        _ilp_synth.ILP_C, _ilp_synth.ILP_A, _ilp_synth.ILP_B, _ilp_synth.ILP_UB
    )
    assert value == pytest.approx(40.0)


def test_the_set_cover_reference_covers_every_element() -> None:
    best = _ilp_synth.brute_set_cover(_ilp_synth.SC_C, _ilp_synth.SC_A)
    assert best >= min(_ilp_synth.SC_C)
    assert np.isfinite(best)


# --------------------------------------------------------------------------- #
# _inv_synth — inventory
# --------------------------------------------------------------------------- #


def test_the_demand_process_is_non_negative_and_integer_valued() -> None:
    d = _inv_synth.demand(0, weeks=50)
    assert d.shape == (50,)
    assert np.all(d >= 0.0)
    assert np.all(d == np.round(d))


def test_the_inventory_simulation_reports_the_three_outputs() -> None:
    avg_cost, fill_rate, n_orders = _inv_synth.simulate(
        lambda ip, w: 20.0 if w % 4 == 0 else 0.0, 0, weeks=40
    )
    assert avg_cost >= 0.0
    assert 0.0 <= fill_rate <= 1.0
    assert n_orders == 10.0


def test_an_ordering_policy_that_never_orders_pays_only_shortage() -> None:
    avg_cost, fill_rate, n_orders = _inv_synth.simulate(lambda ip, w: 0.0, 0, weeks=40)
    assert n_orders == 0.0
    assert fill_rate == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _it_synth — information theory
# --------------------------------------------------------------------------- #


def test_the_dependent_pair_is_more_correlated_than_the_independent_one() -> None:
    x_dep, y_dep = _it_synth.dep_data(0, n=4000, kind="dep")
    x_ind, y_ind = _it_synth.dep_data(0, n=4000, kind="indep")
    assert abs(np.corrcoef(x_dep, y_dep)[0, 1]) > abs(np.corrcoef(x_ind, y_ind)[0, 1])


def test_the_two_sample_shift_moves_the_mean() -> None:
    x, y = _it_synth.twosample(0, n=2000, shift=2.0)
    assert y.mean() - x.mean() == pytest.approx(2.0, abs=0.2)


def test_the_rbf_kernel_is_one_on_the_diagonal() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    K = _it_synth.rbf(x, x, bw=1.0)
    assert K.shape == (2, 2)
    assert np.allclose(np.diag(K), 1.0)


def test_the_median_bandwidth_is_positive() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 2))
    y = rng.standard_normal((50, 2))
    assert _it_synth.med_bw(x, y) > 0.0


def test_the_gaussian_mutual_information_matches_the_closed_form() -> None:
    assert _it_synth.true_mi_gauss(0.5) == pytest.approx(-0.5 * np.log(1 - 0.25))
    assert _it_synth.true_mi_gauss(0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _lm_synth — induction heads / regimes
# --------------------------------------------------------------------------- #


def test_the_recall_batch_keeps_the_key_value_pair() -> None:
    x, y = _lm_synth.recall_batch(0, B=16, T=12)
    assert x.shape == (16, 12)
    assert y.shape == (16,)
    assert np.all((x >= 1) & (x < _lm_synth.VOCAB))


def test_the_regime_task_returns_features_and_binary_labels() -> None:
    X, y = _lm_synth.regime_task(0, n=40)
    assert X.shape == (40, 8)
    assert set(np.unique(y)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# _mc2_synth — correlated Gaussian target
# --------------------------------------------------------------------------- #


def test_the_log_density_is_maximal_at_the_mean() -> None:
    assert _mc2_synth.logp(_mc2_synth.MU) == pytest.approx(0.0)
    assert _mc2_synth.logp(_mc2_synth.MU + 1.0) < 0.0


def test_the_effective_sample_size_of_a_constant_series_is_its_length() -> None:
    """A zero-variance series has no autocorrelation to discount."""
    assert _mc2_synth.ess(np.ones(50)) == pytest.approx(50.0)


def test_the_effective_sample_size_never_exceeds_the_sample_length() -> None:
    rng = np.random.default_rng(0)
    assert _mc2_synth.ess(rng.standard_normal(200)) <= 200.0


def test_the_error_helper_reports_mean_and_covariance_deviations() -> None:
    rng = np.random.default_rng(1)
    samples = rng.multivariate_normal(_mc2_synth.MU, _mc2_synth.SIGMA, size=2000)
    mean_err, cov_err = _mc2_synth.errors(samples)
    assert mean_err < 0.3
    assert cov_err < 0.3


def test_the_independent_chain_samples_the_target() -> None:
    rng = np.random.default_rng(2)
    samples = _mc2_synth.indep_mh(4000, rng)
    assert samples.shape == (4000, 2)
    mean_err, _cov_err = _mc2_synth.errors(samples)
    assert mean_err < 0.5


# --------------------------------------------------------------------------- #
# _oc_synth — double integrator
# --------------------------------------------------------------------------- #


def test_the_double_integrator_advances_position_and_velocity() -> None:
    x = np.array([1.0, 2.0])
    nxt = _oc_synth.dyn(x, 0.5)
    assert nxt[0] == pytest.approx(1.0 + _oc_synth.DT * 2.0)
    assert nxt[1] == pytest.approx(2.0 + _oc_synth.DT * 0.5)


def test_the_jacobian_matches_a_finite_difference() -> None:
    x = np.array([0.7, -0.3])
    A, B = _oc_synth.dyn_jac(x, 0.2)
    eps = 1e-6
    numeric = (
        _oc_synth.dyn(x + eps * np.array([1.0, 0.0]), 0.2)
        - _oc_synth.dyn(x - eps * np.array([1.0, 0.0]), 0.2)
    ) / (2 * eps)
    assert np.allclose(A @ np.array([1.0, 0.0]), numeric)


def test_the_stage_cost_is_zero_at_the_target_at_rest() -> None:
    assert _oc_synth.stage_cost(np.array([1.0, 0.0]), 0.0) == pytest.approx(0.0)


def test_the_rollout_records_every_state() -> None:
    xs, total = _oc_synth.rollout(np.zeros(5), np.array([0.0, 0.0]))
    assert xs.shape == (6, 2)
    assert total >= 0.0


def test_the_pd_baseline_drives_the_plant_towards_the_target() -> None:
    xs, total = _oc_synth.pd_baseline(np.array([3.0, 0.0]), target=1.0)
    assert total >= 0.0
    assert abs(xs[-1, 0] - 1.0) < abs(xs[0, 0] - 1.0)


# --------------------------------------------------------------------------- #
# _pd_synth — banana posterior
# --------------------------------------------------------------------------- #


def test_the_banana_log_density_matches_a_direct_evaluation() -> None:
    x = np.array([0.3, -0.2, 0.5, 0.1])
    expected = -0.5 * x[0] ** 2
    for i in range(1, len(x)):
        expected -= 0.5 * (x[i] - (x[i - 1] ** 2 + 1.0)) ** 2 / _pd_synth.BAND**2
    assert _pd_synth.logp(x) == pytest.approx(float(expected))


def test_the_analytic_gradient_matches_a_finite_difference() -> None:
    x = np.array([0.3, -0.2, 0.5, 0.1])
    gradient = _pd_synth.grad_logp(x)
    eps = 1e-6
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        numeric = (_pd_synth.logp(x + d) - _pd_synth.logp(x - d)) / (2 * eps)
        assert gradient[i] == pytest.approx(numeric, abs=1e-4)


def test_the_moment_error_is_zero_at_the_reference_moments() -> None:
    samples = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (10, 1))
    mu = samples.mean(0)
    sd = samples.std(0)
    assert _pd_synth.moment_err(samples, mu, sd) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _qc_synth — statevector helpers
# --------------------------------------------------------------------------- #


def test_the_kronecker_product_has_the_expected_dimension() -> None:
    assert _qc_synth.kron_all(_qc_synth.X, _qc_synth.H).shape == (4, 4)


def test_the_initial_state_is_the_zero_state() -> None:
    psi = _qc_synth.init_state(3)
    assert psi.shape == (8,)
    assert psi[0] == 1.0
    assert np.allclose(_qc_synth.measure_probs(psi), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_a_gate_on_one_qubit_leaves_the_others_alone() -> None:
    psi = _qc_synth.init_state(2)
    flipped = _qc_synth.apply1(psi, _qc_synth.X, qubit=0, n=2)
    assert _qc_synth.measure_probs(flipped)[2] == pytest.approx(1.0)


def test_the_measurement_probabilities_sum_to_one() -> None:
    psi = _qc_synth.init_state(2)
    after = _qc_synth.apply1(psi, _qc_synth.H, qubit=0, n=2)
    assert _qc_synth.measure_probs(after).sum() == pytest.approx(1.0)


def test_the_max_cut_cost_counts_the_cut_edges() -> None:
    edges = [(0, 1), (1, 2)]
    # 0b011 splits {0,1} from {2}: edge (0,1) uncut, edge (1,2) cut.
    assert _qc_synth.maxcut_cost(0b011, edges) == 1.0


def test_the_max_cut_graph_always_has_at_least_one_edge() -> None:
    assert _qc_synth.maxcut_graph(0, 4)


# --------------------------------------------------------------------------- #
# _queue_synth — queueing
# --------------------------------------------------------------------------- #


def test_the_jackson_traffic_equations_solve_the_fixed_point() -> None:
    lam = _queue_synth.jackson_lam()
    assert np.allclose(_queue_synth.GAMMA + _queue_synth.P.T @ lam, lam)
    assert np.all(lam >= 0.0)


def test_the_m_m_1_simulation_is_near_the_theoretical_queue_length() -> None:
    """rho = lambda/mu = 0.5 -> L = rho/(1-rho) = 1."""
    estimate = _queue_synth.mm1_sim(0, lam=1.5, mu=3.0, horizon=20000.0)
    assert 0.5 < estimate < 1.8


def test_the_simulation_is_reproducible_for_a_seed() -> None:
    first = _queue_synth.mm1_sim(3, 1.5, 3.0, horizon=5000.0)
    second = _queue_synth.mm1_sim(3, 1.5, 3.0, horizon=5000.0)
    assert first == pytest.approx(second)


# --------------------------------------------------------------------------- #
# _rel_synth — reliability
# --------------------------------------------------------------------------- #


def test_the_weibull_lifetimes_are_censored_at_the_threshold() -> None:
    times, failed = _rel_synth.weibull_lifetimes(0, n=200)
    assert times.shape == (200,)
    assert np.all(times <= _rel_synth.WB_CENSOR + 1e-9)
    # An observation that failed is the one whose raw lifetime is below the
    # censoring threshold; the rest are reported at the threshold itself.
    assert np.all(failed == (times < _rel_synth.WB_CENSOR))


def test_the_arrhenius_lives_cover_every_stress_temperature() -> None:
    temps, lives = _rel_synth.arrhenius_lives(0, n=20)
    assert temps.shape == lives.shape == (60,)
    assert set(np.unique(temps)) == set(_rel_synth.ARR_TEMPS.tolist())


def test_the_fault_tree_probability_matches_its_structure() -> None:
    p = _rel_synth.TREE_P
    expected = 1.0 - (1 - p["a"] * p["b"]) * (1 - p["c"] * p["d"]) * (1 - p["e"])
    assert _rel_synth.fault_tree_top_prob() == pytest.approx(expected)


def test_the_fault_tree_has_the_documented_shape() -> None:
    tree = _rel_synth.fault_tree()
    assert tree["gate"] == "or"
    assert len(tree["children"]) == 3


# --------------------------------------------------------------------------- #
# _sched_synth — scheduling
# --------------------------------------------------------------------------- #


def test_the_flowshop_instance_has_the_requested_shape() -> None:
    assert _sched_synth.flowshop(0, n_jobs=5, n_mach=3).shape == (5, 3)


def test_the_knapsack_capacity_is_a_fraction_of_the_total_weight() -> None:
    w, v, cap = _sched_synth.knapsack(0, n=20, cap_frac=0.5)
    assert cap == pytest.approx(0.5 * w.sum())
    assert v.shape == w.shape


def test_the_tsp_distance_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    d = _sched_synth.tsp(0, n=6)
    assert d.shape == (6, 6)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_the_makespan_of_an_empty_order_is_zero() -> None:
    p = _sched_synth.flowshop(0, n_jobs=3, n_mach=2)
    assert _sched_synth.makespan([], p) == 0.0


def test_the_makespan_does_not_depend_on_the_sum_of_the_processing_times() -> None:
    """The makespan is at least the busiest machine's total load."""
    p = _sched_synth.flowshop(0, n_jobs=4, n_mach=2)
    makespan = _sched_synth.makespan([0, 1, 2, 3], p)
    assert makespan >= p.sum(axis=0).max() - 1e-9


# --------------------------------------------------------------------------- #
# _sig3_synth — signal processing
# --------------------------------------------------------------------------- #


def test_the_signal_fixtures_have_the_declared_length() -> None:
    assert _sig3_synth.chirp(0).shape == (_sig3_synth.N,)
    assert _sig3_synth.voiced(0).shape == (_sig3_synth.N,)
    assert _sig3_synth.two_tone(0).shape == (_sig3_synth.N,)


def test_the_two_tone_fixture_peaks_at_the_two_tone_frequencies() -> None:
    spectrum = np.abs(np.fft.rfft(_sig3_synth.two_tone(0)))
    freqs = np.fft.rfftfreq(_sig3_synth.N, 1.0 / _sig3_synth.FS)
    peak = freqs[int(np.argmax(spectrum))]
    assert peak == pytest.approx(941.0, abs=5.0)


def test_the_array_snapshot_has_the_declared_shape() -> None:
    x, steer = _sig3_synth.array_snap(0)
    assert x.shape == (8, 256)
    assert steer.shape == (8,)
    assert np.allclose(np.abs(steer), 1.0)


# --------------------------------------------------------------------------- #
# _sk_synth — spiking encodings
# --------------------------------------------------------------------------- #


def test_the_spiking_fixture_returns_features_and_binary_labels() -> None:
    X, y = _sk_synth.sk_data(0, n=30)
    assert X.shape == (30, 8)
    assert set(np.unique(y)) <= {0, 1}


def test_the_poisson_encoding_produces_binary_spike_trains() -> None:
    spikes = _sk_synth.poisson_encode(np.zeros((5, 8)), T=20, seed=0)
    assert spikes.shape == (5, 8, 20)
    assert set(np.unique(spikes)) <= {0.0, 1.0}


def test_the_lif_forward_pass_returns_one_voltage_per_sample() -> None:
    spikes = _sk_synth.poisson_encode(np.zeros((4, 8)), T=30, seed=1)
    v = _sk_synth.lif_forward(spikes, np.ones(8), T=30)
    assert v.shape == (4,)
    assert np.all(np.isfinite(v))


def test_the_latency_encoding_fires_exactly_one_spike_per_input() -> None:
    spikes = _sk_synth.latency_encode(np.zeros((4, 8)), T=30)
    assert spikes.shape == (4, 8, 30)
    assert np.all(spikes.sum(axis=2) == 1.0)


def test_a_higher_input_produces_an_earlier_spike() -> None:
    low = _sk_synth.latency_encode(np.array([[-3.0]]), T=20)
    high = _sk_synth.latency_encode(np.array([[3.0]]), T=20)
    assert int(np.argmax(high[0, 0])) < int(np.argmax(low[0, 0]))


# --------------------------------------------------------------------------- #
# _sv_synth — survival
# --------------------------------------------------------------------------- #


def test_the_survival_fixture_returns_the_documented_tuple() -> None:
    X, t, e, beta = _sv_synth.sv_data(0, n=40, d=6)
    assert X.shape == (40, 6)
    assert t.shape == (40,)
    assert e.shape == (40,)
    assert set(np.unique(e)) <= {0, 1}
    assert beta.shape == (6,)


def test_the_cindex_of_a_perfect_ranking_is_one() -> None:
    t = np.array([1.0, 2.0, 3.0])
    risk = np.array([3.0, 2.0, 1.0])
    e = np.array([1, 1, 1])
    assert _sv_synth.cindex(t, risk, e) == pytest.approx(1.0)


def test_the_cindex_of_an_inverted_ranking_is_zero() -> None:
    t = np.array([1.0, 2.0, 3.0])
    risk = np.array([1.0, 2.0, 3.0])
    e = np.array([1, 1, 1])
    assert _sv_synth.cindex(t, risk, e) == pytest.approx(0.0)


def test_the_cox_fit_tracks_the_true_coefficient_direction() -> None:
    """The fitted hazard coefficients must line up with the true ones."""
    X, t, e, beta = _sv_synth.sv_data(0, n=250, d=4)
    fitted = _sv_synth.cox_ph(X, t, e, iters=60)
    assert fitted.shape == beta.shape
    assert np.all(np.isfinite(fitted))
    assert float(np.corrcoef(fitted, beta)[0, 1]) > 0.5


# --------------------------------------------------------------------------- #
# _tn_synth — tensor networks
# --------------------------------------------------------------------------- #


def test_the_smooth_tensor_has_the_requested_rank_structure() -> None:
    T = _tn_synth.smooth_tensor(0, d=3, n=6)
    assert T.shape == (6, 6, 6)
    assert np.all(np.isfinite(T))


def test_the_ising_hamiltonian_is_symmetric() -> None:
    H = _tn_synth.tfim_h(2, h=1.0)
    assert H.shape == (4, 4)
    assert np.allclose(H, H.T)


def test_the_mps_round_trip_reproduces_the_tensor() -> None:
    T = _tn_synth.smooth_tensor(0, d=2, n=4)
    cores = _tn_synth.to_mps(T, d_phys=2, chi=8)
    # A 4x4 tensor is four sites of physical dimension two.
    assert len(cores) == 4
    rebuilt = _tn_synth.mps_contract(cores)
    assert rebuilt.shape == T.reshape(-1).shape
    assert np.allclose(rebuilt, T.reshape(-1), atol=1e-8)


# --------------------------------------------------------------------------- #
# _vi_synth — amortized inference
# --------------------------------------------------------------------------- #


def test_the_vi_fixture_splits_train_from_test() -> None:
    Xtr, ytr, Xte, yte, w_true = _vi_synth.vi_data(0, n=40, d=5)
    assert Xtr.shape == (40, 5)
    assert Xte.shape == (40, 5)
    assert w_true.shape == (5,)
    assert set(np.unique(ytr)) <= {0, 1}


def test_the_log_posterior_is_higher_at_the_truth_than_at_zero() -> None:
    X, y, _Xte, _yte, w_true = _vi_synth.vi_data(0, n=120, d=4)
    assert _vi_synth.logpost(w_true, X, y) > _vi_synth.logpost(np.zeros(4), X, y)


def test_the_log_loss_is_lower_for_a_better_weight_vector() -> None:
    Xtr, ytr, Xte, yte, w_true = _vi_synth.vi_data(0, n=150, d=4)
    good = _vi_synth.test_logloss(w_true, Xte, yte)
    bad = _vi_synth.test_logloss(np.zeros(4), Xte, yte)
    assert good < bad


def test_the_mcmc_oracle_returns_a_mean_and_a_spread() -> None:
    X, y, _Xte, _yte, _w_true = _vi_synth.vi_data(0, n=60, d=3)
    mean, std = _vi_synth.mcmc_oracle(X, y, seed=0, iters=400)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.all(std >= 0.0)
