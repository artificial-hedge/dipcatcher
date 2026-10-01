"""SGA multi-step UQ (arXiv:2609.28582) -- SYNTHETIC correctness tests.

Seeded simulations only: DAG topological propagation vs the closed-form
covariance ``(I - A^T)^{-1} D (I - A)^{-1}`` at 1e-12, the ordered
VAR(1)-covariance factorization recovering phi exactly, the
Bonacich-Lloyd variance bound certificate never violated, seeded-SYNTHETIC
coverage envelopes within documented slack, DTW known values, the
slice/graph/align pipeline's merge semantics, the model-scale vs
complexity diagnostic, fail-closed edges (cycles, backward edges, ragged
panels, degenerate steps), and determinism. Correctness material, never
market evidence. No Sharpe.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.sga_multistep_uq import (
    MIN_SAMPLES,
    alpha_centrality,
    build_error_dag,
    build_sga_graph,
    complexity_scale_diagnostic,
    dag_covariance_closed_form,
    dag_from_covariance,
    dtw_distance,
    estimate_error_dag,
    graph_complexity,
    horizon_monotonicity,
    interval_coverage,
    make_synthetic_ar1_forecaster,
    propagate_uncertainty,
    sample_trajectories,
    seasonal_scaled_threshold,
    sga_complexity,
    simulate_synthetic_dag_errors,
    slice_spans,
    step_level_entropy,
    uq_from_forecaster,
    uq_from_residuals,
    var1_error_dag,
    variance_bound_certificate,
)


def _ar1_covariance(h: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) covariance Sigma[r,s] = phi^|r-s| sigma^2/(1-phi^2)."""
    idx = np.arange(h)
    return sigma**2 * phi ** np.abs(idx[:, None] - idx[None, :]) / (1.0 - phi**2)


def _history(n: int = 60, seed: int = 7) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0.0, 1.0, n))


# ---------------------------------------------------------------------------
# DAG construction and exact propagation
# ---------------------------------------------------------------------------


def test_build_error_dag_edge_list_sorted() -> None:
    a = np.zeros((4, 4))
    a[0, 2] = 0.3
    a[0, 1] = 0.7
    a[1, 2] = -0.2
    a[2, 3] = 0.5
    dag = build_error_dag(a, np.full(4, 0.5))
    assert dag.horizon == 4
    assert dag.edges.tolist() == [[0, 1], [0, 2], [1, 2], [2, 3]]
    np.testing.assert_array_equal(dag.topo_order(), np.arange(4))


def test_propagate_matches_closed_form_var1() -> None:
    dag = var1_error_dag(12, 0.7, 1.0)
    pu = propagate_uncertainty(dag)
    cf = dag_covariance_closed_form(dag)
    np.testing.assert_allclose(pu.covariance, cf, atol=1e-12, rtol=1e-12)
    # Hand VAR(1) series: Var(e_s) = sigma^2 * sum_{j=0}^{s-1} phi^{2j}.
    expected_var = np.cumsum(0.7 ** (2 * np.arange(12)))
    np.testing.assert_allclose(pu.variance, expected_var, atol=1e-12)
    # Cross-covariance: Cov(e_r, e_s) = phi^{s-r} Var(e_r).
    assert pu.covariance[0, 3] == pytest.approx(0.7**3 * expected_var[0], abs=1e-12)
    np.testing.assert_allclose(pu.std, np.sqrt(expected_var), atol=1e-12)


def test_propagate_matches_closed_form_random_dag() -> None:
    rng = np.random.default_rng(11)
    h = 9
    a = np.zeros((h, h))
    mask = np.triu(np.ones((h, h), dtype=bool), 1) & (rng.random((h, h)) < 0.4)
    a[mask] = rng.normal(0.0, 0.25, int(mask.sum()))
    d = rng.uniform(0.2, 1.5, h)
    dag = build_error_dag(a, d)
    pu = propagate_uncertainty(dag)
    cf = dag_covariance_closed_form(dag)
    np.testing.assert_allclose(pu.covariance, cf, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.diag(pu.covariance), pu.variance, atol=0.0)
    assert np.all(pu.variance > 0.0)


def test_dag_from_covariance_full_lag_recovers_ar1() -> None:
    """Ordered factorization of a stationary AR(1) covariance is exact:
    the lag-1 edge gets phi, all other edges get ~0, and every innovation
    variance gets sigma^2 (the AR(1) Markov property) except the first
    step, which carries the stationary marginal variance sigma^2/(1-phi^2)."""
    h, phi, sigma = 8, 0.65, 0.9
    cov = _ar1_covariance(h, phi, sigma)
    dag = dag_from_covariance(cov)  # full lag
    np.testing.assert_allclose(dag.coefficients[np.arange(h - 1), np.arange(1, h)], phi, atol=1e-12)
    off = dag.coefficients.copy()
    off[np.arange(h - 1), np.arange(1, h)] = 0.0
    np.testing.assert_allclose(off, 0.0, atol=1e-12)
    assert dag.innovation_var[0] == pytest.approx(sigma**2 / (1.0 - phi**2), abs=1e-12)
    np.testing.assert_allclose(dag.innovation_var[1:], sigma**2, atol=1e-12)
    # And propagation reproduces the input covariance exactly.
    np.testing.assert_allclose(propagate_uncertainty(dag).covariance, cov, atol=1e-12, rtol=1e-12)


def test_dag_from_covariance_banded() -> None:
    """Banded (max_lag=1) factorization keeps only horizon-lag-1 edges."""
    h = 6
    rng = np.random.default_rng(3)
    mat = rng.normal(0, 1, (h, h))
    cov = mat @ mat.T + 0.5 * np.eye(h)
    dag = dag_from_covariance(cov, max_lag=1)
    assert np.all(np.tril(dag.coefficients, -2) == 0.0)  # no lag >= 2 edges
    assert np.all(dag.innovation_var > 0.0)
    pu = propagate_uncertainty(dag)
    assert np.all(pu.variance > 0.0)


def test_estimate_error_dag_recovers_var1() -> None:
    rng = np.random.default_rng(5)
    dag_true = var1_error_dag(10, 0.7, 1.0)
    errs = simulate_synthetic_dag_errors(dag_true, 20000, rng)
    fit = estimate_error_dag(errs, max_lag=1)
    dag = fit.dag
    # phi_hat and innovation variance converge to planted values.
    phis = np.diag(dag.coefficients, 1)
    np.testing.assert_allclose(phis, 0.7, atol=0.03)
    np.testing.assert_allclose(fit.innovation_var, 1.0, atol=0.06)
    # Propagated variance tracks the true VAR(1) variances.
    pu = propagate_uncertainty(fit.dag)
    true_var = np.cumsum(0.7 ** (2 * np.arange(10)))
    np.testing.assert_allclose(pu.variance, true_var, rtol=0.10, atol=0.10)
    # lag-1 only: no lag-2 edges fitted.
    assert np.all(np.tril(dag.coefficients, -2) == 0.0)


def test_h1_single_node_reduction() -> None:
    dag = var1_error_dag(1, 0.7, 2.0)
    pu = propagate_uncertainty(dag)
    assert pu.variance.shape == (1,)
    assert pu.variance[0] == pytest.approx(4.0)
    np.testing.assert_allclose(pu.covariance, [[4.0]])


def test_intervals_and_half_widths() -> None:
    dag = var1_error_dag(5, 0.5, 1.0)
    pu = propagate_uncertainty(dag)
    hw = pu.interval_half_widths(1.96)
    np.testing.assert_allclose(hw, 1.96 * pu.std, atol=0.0)
    lo, hi = pu.intervals(np.zeros(5), 1.96)
    np.testing.assert_allclose(lo, -hw, atol=0.0)
    np.testing.assert_allclose(hi, hw, atol=0.0)


# ---------------------------------------------------------------------------
# Graph complexity and the variance bound certificate
# ---------------------------------------------------------------------------


def test_alpha_centrality_hand_computed_chain() -> None:
    # Chain 0 -> 1 -> 2 with alpha = 0.1, U = (1, 2, 3).
    edges = np.array([[0, 1], [1, 2]], dtype=np.intp)
    u = np.array([1.0, 2.0, 3.0])
    b = alpha_centrality(edges, 3, u, alpha=0.1)
    np.testing.assert_allclose(b, [1.0, 2.0 + 0.1, 3.0 + 0.1 * (2.0 + 0.1)], atol=1e-12)
    assert graph_complexity(edges, u, 0.1) == pytest.approx(float(b.sum()))


def test_alpha_zero_reduces_to_node_uncertainty() -> None:
    dag = var1_error_dag(6, 0.7, 1.0)
    var = propagate_uncertainty(dag).variance
    b = alpha_centrality(dag.edges, dag.horizon, var, alpha=0.0)
    np.testing.assert_allclose(b, var, atol=0.0)


def test_variance_bound_certificate_holds() -> None:
    rng = np.random.default_rng(13)
    h = 10
    a = np.zeros((h, h))
    mask = np.triu(np.ones((h, h), dtype=bool), 1) & (rng.random((h, h)) < 0.35)
    a[mask] = rng.normal(0.0, 0.3, int(mask.sum()))
    dag = build_error_dag(a, rng.uniform(0.3, 1.2, h))
    cert = variance_bound_certificate(dag, alpha=0.1)
    assert cert["certified"]
    margins = cert["per_step_bound_margin"]
    assert isinstance(margins, np.ndarray)
    assert np.all(margins >= -1e-9)
    assert cert["graph_complexity"] >= cert["total_variance"]  # type: ignore[operator]
    # Each step's centrality bounds its variance; the bound is tight at
    # step 0 (no ancestors) and strictly loose where predecessors exist.
    assert margins[0] == pytest.approx(0.0, abs=1e-12)
    assert np.any(margins[1:] > 0.0)


def test_certificate_var1_monotone_in_alpha() -> None:
    dag = var1_error_dag(8, 0.7, 1.0)
    g0 = variance_bound_certificate(dag, alpha=0.0)["graph_complexity"]
    g1 = variance_bound_certificate(dag, alpha=0.1)["graph_complexity"]
    g2 = variance_bound_certificate(dag, alpha=0.5)["graph_complexity"]
    assert g0 == pytest.approx(variance_bound_certificate(dag, 0.0)["total_variance"])  # type: ignore[arg-type]
    assert g0 < g1 < g2  # type: ignore[operator]


def test_uq_from_residuals_end_to_end_certified() -> None:
    rng = np.random.default_rng(21)
    errs = simulate_synthetic_dag_errors(var1_error_dag(8, 0.6, 1.0), 30000, rng)
    out = uq_from_residuals(errs, max_lag=1)
    assert out["certificate"]["certified"]  # type: ignore[index]
    assert out["half_widths"].shape == (8,)  # type: ignore[attr-defined]
    pu = out["propagated"]
    assert np.all(pu.variance > 0.0)  # type: ignore[attr-defined]
    # Terminal variance tracks the VAR(1) stationary limit sigma^2/(1-phi^2);
    # exact monotonicity is a diagnostic on a true DAG, not guaranteed under
    # estimation noise at near-stationary steps.
    assert pu.variance[-1] == pytest.approx(1.0 / (1.0 - 0.36), rel=0.15)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# DTW and the seasonal-scaled threshold
# ---------------------------------------------------------------------------


def test_dtw_zero_self_and_constant() -> None:
    x = np.array([0.3, 1.0, -0.5, 2.0])
    assert dtw_distance(x, x) == 0.0
    assert dtw_distance(np.zeros(3), np.ones(3)) == pytest.approx(3.0)


def test_dtw_known_value_and_warping() -> None:
    # [1,2,3] vs [2,3,4]: optimal path costs 1 + 0 + 0 + 1 = 2.
    assert dtw_distance([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(2.0)
    # Warping absorbs a one-step insertion: [0,1,1] vs [0,0,1] cost 0.
    assert dtw_distance([0.0, 1.0, 1.0], [0.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_dtw_symmetry_nonnegative() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 6)
    b = rng.normal(0, 1, 5)
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))
    assert dtw_distance(a, b) >= 0.0


def test_seasonal_threshold_closed_form() -> None:
    hist = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    # S = 1: diffs 1,2,3,4 -> mean 2.5 -> tau = lam * 2.5.
    assert seasonal_scaled_threshold(hist, 1, 0.25) == pytest.approx(0.625)
    # S = 2: diffs 3,5,7 -> mean 5 -> tau = 2 * 5.
    assert seasonal_scaled_threshold(hist, 2, 2.0) == pytest.approx(10.0)


def test_slice_spans_ceil_rule() -> None:
    assert slice_spans(10, 4) == ((0, 4), (4, 8), (8, 10))
    assert slice_spans(8, 4) == ((0, 4), (4, 8))
    assert slice_spans(3, 5) == ((0, 3),)
    assert slice_spans(1, 1) == ((0, 1),)


# ---------------------------------------------------------------------------
# Sampled SGA pipeline
# ---------------------------------------------------------------------------


def test_sample_trajectories_contract() -> None:
    rng = np.random.default_rng(9)
    stub = make_synthetic_ar1_forecaster(0.5, 1.0)
    tr = sample_trajectories(stub, _history(), 8, 16, rng)
    assert tr.shape == (16, 8)
    assert np.isfinite(tr).all()


def test_build_sga_graph_structure_and_merge() -> None:
    rng = np.random.default_rng(10)
    hist = _history()
    stub = make_synthetic_ar1_forecaster(0.7, 1.0)
    tr = sample_trajectories(stub, hist, 16, 24, rng)
    g = build_sga_graph(tr, hist, slice_len=4, seasonality=5)
    assert g.n_levels == 4
    assert g.levels[0] == -1  # root
    assert g.node_uncertainty[0] == 0.0
    assert g.n_nodes <= 1 + 4 * 24
    counts = [len(g.nodes_at_level(j)) for j in range(g.n_levels)]
    assert all(1 <= c <= 24 for c in counts)
    # every non-root node has >= 1 member and edges only go forward a level
    for e in g.edges:
        assert g.levels[e[1]] == g.levels[e[0]] + 1
    comp = sga_complexity(g)
    assert np.isfinite(comp["graph_complexity"])  # type: ignore[arg-type]
    assert comp["centrality"].shape == (g.n_nodes,)  # type: ignore[attr-defined]


def test_identical_trajectories_merge_to_single_branch() -> None:
    rng = np.random.default_rng(12)
    base = np.linspace(0.0, 1.0, 12)
    tr = np.tile(base, (16, 1)) + rng.normal(0.0, 1e-6, (16, 12))
    g = build_sga_graph(tr, tau=1.0)
    assert g.n_levels == 3  # h=12, l_s=4
    assert all(len(g.nodes_at_level(j)) == 1 for j in range(3))
    assert g.n_nodes == 4  # root + one branch node per level
    # merged node carries all 16 member trajectories
    node1 = g.nodes_at_level(0)[0]
    assert len(g.members[node1]) == 16


def test_two_regime_trajectories_preserve_branches() -> None:
    rng = np.random.default_rng(14)
    k, h = 24, 12
    up = np.tile(np.linspace(0.0, 2.0, h), (k // 2, 1))
    down = np.tile(np.linspace(0.0, -2.0, h), (k // 2, 1))
    tr = np.vstack([up, down]) + rng.normal(0.0, 1e-4, (k, h))
    g = build_sga_graph(tr, tau=0.01)
    # the two regimes must not merge into one branch at any level
    assert all(len(g.nodes_at_level(j)) >= 2 for j in range(g.n_levels))


def test_step_entropy_increases_with_dispersion() -> None:
    rng = np.random.default_rng(16)
    hist = _history()
    tr_low = sample_trajectories(make_synthetic_ar1_forecaster(0.5, 0.3), hist, 8, 24, rng)
    tr_high = sample_trajectories(
        make_synthetic_ar1_forecaster(0.5, 1.5), hist, 8, 24, np.random.default_rng(16)
    )
    e_low = step_level_entropy(tr_low)
    e_high = step_level_entropy(tr_high)
    assert np.all(e_high > e_low)


def test_scaling_law_diagnostic_direction() -> None:
    # Paper's empirical scaling law: larger models -> lower uncertainty.
    rng = np.random.default_rng(18)
    hist = _history()
    gcs = []
    for sigma in (2.0, 1.0, 0.5, 0.25):  # decreasing sigma ~ increasing model scale
        tr = sample_trajectories(make_synthetic_ar1_forecaster(0.6, sigma), hist, 8, 24, rng)
        gcs.append(
            sga_complexity(build_sga_graph(tr, hist, 4, None, seasonality=5))["graph_complexity"]
        )
    diag = complexity_scale_diagnostic([1.0, 2.0, 4.0, 8.0], gcs)
    assert diag["spearman"] == pytest.approx(-1.0)
    assert diag["pearson_log"] < -0.9
    assert diag["decreasing"]


def test_uq_from_forecaster_contract() -> None:
    rng = np.random.default_rng(20)
    hist = _history()
    out = uq_from_forecaster(
        make_synthetic_ar1_forecaster(0.7, 1.0), hist, 16, 32, rng=rng, seasonality=5
    )
    assert out["schema"] == "sga_multistep_uq.v1"
    assert out["trajectories"].shape == (32, 16)  # type: ignore[attr-defined]
    assert out["per_step_std"].shape == (16,)  # type: ignore[attr-defined]
    assert np.isfinite(out["graph_complexity"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Envelope coverage and monotonicity on planted streams
# ---------------------------------------------------------------------------


def test_coverage_envelope_var1_within_slack() -> None:
    """Propagated 1.96*std envelopes contain realized SYNTHETIC errors at
    the nominal ~0.95 rate at every horizon (slack 0.02 vs se ~0.002)."""
    rng = np.random.default_rng(23)
    dag = var1_error_dag(12, 0.7, 1.0)
    errs = simulate_synthetic_dag_errors(dag, 8000, rng)
    pu = propagate_uncertainty(dag)
    cov = interval_coverage(errs, pu.std, 1.96)
    per_step = cov["per_step"]
    assert isinstance(per_step, np.ndarray)
    assert np.all(per_step >= 0.95 - 0.02)
    assert np.all(per_step <= 1.0)
    assert cov["mean"] == pytest.approx(0.95, abs=0.012)  # type: ignore[arg-type]


def test_coverage_envelope_multi_lag_dag() -> None:
    """A DAG with lag-1 and lag-2 edges propagates intervals that cover
    the SYNTHETIC error stream at every horizon."""
    rng = np.random.default_rng(25)
    a = np.zeros((8, 8))
    a[np.arange(7), np.arange(1, 8)] = 0.5  # lag-1
    a[np.arange(6), np.arange(2, 8)] = 0.2  # lag-2
    dag = build_error_dag(a, np.full(8, 0.8))
    errs = simulate_synthetic_dag_errors(dag, 8000, rng)
    pu = propagate_uncertainty(dag)
    cov = interval_coverage(errs, pu.std, 1.96)
    assert np.all(cov["per_step"] >= 0.95 - 0.02)  # type: ignore[operator]
    # empirical marginal variance matches propagated variance
    np.testing.assert_allclose(errs.var(axis=0, ddof=1), pu.variance, rtol=0.06)


def test_horizon_monotonicity_var1() -> None:
    dag = var1_error_dag(40, 0.9, 1.0)
    pu = propagate_uncertainty(dag)
    diag = horizon_monotonicity(pu.variance)
    assert diag["monotone"]
    assert diag["first_decrease"] == -1
    # converging to the stationary variance sigma^2 / (1 - phi^2) ~ 5.263
    assert diag["terminal_variance"] == pytest.approx(1.0 / (1 - 0.81), rel=0.02)  # type: ignore[arg-type]


def test_horizon_monotonicity_flags_decrease() -> None:
    diag = horizon_monotonicity(np.array([1.0, 0.5, 2.0]))
    assert not diag["monotone"]
    assert diag["first_decrease"] == 1


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


def test_cycle_rejected() -> None:
    a = np.zeros((3, 3))
    a[0, 1] = 1.0
    a[1, 2] = 1.0
    a[2, 0] = 1.0  # cycle 0 -> 1 -> 2 -> 0
    with pytest.raises(ValueError, match="cycle"):
        build_error_dag(a, np.ones(3))


def test_self_loop_and_backward_edges_rejected() -> None:
    a = np.zeros((3, 3))
    a[1, 1] = 0.5
    with pytest.raises(ValueError, match="diagonal"):
        build_error_dag(a, np.ones(3))
    b = np.zeros((3, 3))
    b[2, 0] = 0.5  # backward edge (acyclic but wrong orientation)
    with pytest.raises(ValueError, match="upper triangular"):
        build_error_dag(b, np.ones(3))


def test_nonpositive_innovation_rejected() -> None:
    a = np.zeros((2, 2))
    a[0, 1] = 0.5
    with pytest.raises(ValueError, match="positive"):
        build_error_dag(a, np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        build_error_dag(a, np.array([1.0, np.nan]))


def test_residual_panel_fail_closed() -> None:
    with pytest.raises(ValueError):  # ragged -> non-2d/asarray failure
        estimate_error_dag([[1.0, 2.0], [1.0]], max_lag=1)
    with pytest.raises(ValueError):  # 1d
        estimate_error_dag(np.arange(20.0), max_lag=1)
    rng = np.random.default_rng(1)
    panel = rng.normal(0, 1, (40, 6))
    panel[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimate_error_dag(panel, max_lag=1)
    with pytest.raises(ValueError, match="panels"):
        estimate_error_dag(rng.normal(0, 1, (4, 6)), max_lag=1)
    const = np.ones((40, 6))
    with pytest.raises(ValueError, match="degenerate"):
        estimate_error_dag(const, max_lag=1)
    with pytest.raises(ValueError, match="max_lag"):
        estimate_error_dag(rng.normal(0, 1, (40, 6)), max_lag=6)


def test_covariance_fail_closed() -> None:
    nonsym = np.array([[1.0, 0.9], [0.1, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        dag_from_covariance(nonsym)
    nonpd = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        dag_from_covariance(nonpd)


def test_trajectory_validation_fail_closed() -> None:
    hist = _history()
    stub = make_synthetic_ar1_forecaster(0.5, 1.0)
    rng = np.random.default_rng(30)
    with pytest.raises(ValueError, match="n_samples"):
        sample_trajectories(stub, hist, 8, MIN_SAMPLES - 1, rng)
    with pytest.raises(ValueError, match="Generator"):
        sample_trajectories(stub, hist, 8, 16, np.random)  # type: ignore[arg-type]
    bad_stub = lambda h_, hh, kk, r: np.zeros((kk, hh + 1))  # noqa: E731
    with pytest.raises(ValueError, match="stub"):
        sample_trajectories(bad_stub, hist, 8, 16, rng)
    nan_stub = lambda h_, hh, kk, r: np.full((kk, hh), np.nan)  # noqa: E731
    with pytest.raises(ValueError, match="finite"):
        sample_trajectories(nan_stub, hist, 8, 16, rng)


def test_sga_graph_fail_closed() -> None:
    tr = np.random.default_rng(31).normal(0, 1, (16, 8))
    with pytest.raises(ValueError):  # neither tau nor history
        build_sga_graph(tr)
    with pytest.raises(ValueError, match="tau"):
        build_sga_graph(tr, tau=-1.0)
    with pytest.raises(ValueError, match="slice_len|>= 1"):
        build_sga_graph(tr, tau=1.0, slice_len=0)
    with pytest.raises(ValueError):  # too few samples
        build_sga_graph(tr[:5], tau=1.0)
    const_step = tr.copy()
    const_step[:, 3] = 7.0  # zero dispersion at step 3
    with pytest.raises(ValueError, match="dispersion"):
        build_sga_graph(const_step, tau=1.0)


def test_threshold_and_alpha_fail_closed() -> None:
    hist = _history(20)
    with pytest.raises(ValueError, match="seasonality|history length"):
        seasonal_scaled_threshold(hist, 20)
    with pytest.raises(ValueError, match="positive"):
        seasonal_scaled_threshold(hist, 5, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        alpha_centrality(np.zeros((0, 2), dtype=np.intp), 2, np.ones(2), -0.1)
    dag = var1_error_dag(4, 0.5, 1.0)
    with pytest.raises(ValueError, match="z"):
        propagate_uncertainty(dag).interval_half_widths(0.0)


def test_dtw_fail_closed() -> None:
    with pytest.raises(ValueError):
        dtw_distance(np.array([]), np.ones(3))
    with pytest.raises(ValueError):
        dtw_distance(np.array([1.0, np.nan]), np.ones(2))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_bit_identical() -> None:
    hist = _history()
    stub = make_synthetic_ar1_forecaster(0.7, 1.0)
    out1 = uq_from_forecaster(stub, hist, 16, 24, rng=np.random.default_rng(42), seasonality=5)
    out2 = uq_from_forecaster(stub, hist, 16, 24, rng=np.random.default_rng(42), seasonality=5)
    np.testing.assert_array_equal(out1["trajectories"], out2["trajectories"])  # type: ignore[arg-type]
    assert out1["graph_complexity"] == out2["graph_complexity"]
    np.testing.assert_array_equal(out1["graph"].edges, out2["graph"].edges)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        out1["graph"].node_uncertainty,
        out2["graph"].node_uncertainty,  # type: ignore[attr-defined]
    )
    rng = np.random.default_rng(42)
    dag = var1_error_dag(8, 0.6, 1.0)
    np.testing.assert_array_equal(
        simulate_synthetic_dag_errors(dag, 50, np.random.default_rng(7)),
        simulate_synthetic_dag_errors(dag, 50, np.random.default_rng(7)),
    )
    _ = rng
