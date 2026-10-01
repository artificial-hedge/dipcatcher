"""Wave 13 lane A5 — PICPIs (Yang, Huang, Hou, Imbens & Jordan 2026, arXiv:2609.25388).

All data here is seeded SYNTHETIC (correctness of the construction and its
guarantees), never market evidence. Assertion tolerances are generous where
they document sampling noise; determinism is pinned by PCG64 seeds.
"""

import itertools

import numpy as np
import pytest

from quant_fund.metrics.picpi import (
    MAX_GRID_BINS,
    MulticlassPicpi,
    _min_weight_cover,
    bench_picpi,
    bench_picpi_width_rate,
    disjoint_calibration,
    fit_picpi,
    fit_picpi_empirical,
    fit_picpi_multiclass,
    holdout_self_consistency,
    naive_bin_acceptance,
    synthetic_picpi_data,
)


def _oracle_multiclass(
    n_cal: int, n_test: int, seed: int, n_classes: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Oracle-probability multiclass fixture: q ~ Dirichlet(1), Y ~ Cat(q).

    With p_g = q_g the one-vs-rest strata are exactly self-consistent:
    E[1{Y=g} | q_g in I] = E[q_g | q_g in I] in I, so every positive-mass
    interval is a population PICPI and the Theorem 4.5 certificates apply.
    """
    rng = np.random.default_rng(seed)
    q_cal = rng.dirichlet(np.ones(n_classes), size=n_cal)
    lab_cal = (rng.random(n_cal)[:, None] > np.cumsum(q_cal, axis=1)).sum(axis=1)
    q_test = rng.dirichlet(np.ones(n_classes), size=n_test)
    lab_test = (rng.random(n_test)[:, None] > np.cumsum(q_test, axis=1)).sum(axis=1)
    # Floating-point cumsum can end slightly below 1; clip the rare overflow.
    lab_cal = np.minimum(lab_cal, n_classes - 1)
    lab_test = np.minimum(lab_test, n_classes - 1)
    return q_cal, lab_cal, q_test, lab_test


# ---------------------------------------------------------------------------
# fail-closed input validation
# ---------------------------------------------------------------------------


def test_fit_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_picpi(np.array([0.1, 0.2]), np.array([0.0]))


def test_fit_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_picpi(np.array([0.1, np.nan]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        fit_picpi(np.array([0.1, 0.2]), np.array([0.0, np.inf]))


def test_fit_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="outcome_range"):
        fit_picpi(np.array([0.1, 1.2]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="outcome_range"):
        fit_picpi(np.array([0.1, 0.2]), np.array([-0.5, 1.0]))


def test_fit_rejects_empty_calibration() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_picpi(np.zeros(0), np.zeros(0))


def test_fit_rejects_bad_delta_and_num_bins() -> None:
    p, y = np.array([0.2, 0.8]), np.array([0.0, 1.0])
    for delta in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="delta"):
            fit_picpi(p, y, delta=delta)
    for k in (0, -3):
        with pytest.raises(ValueError, match="num_bins"):
            fit_picpi(p, y, num_bins=k)
    with pytest.raises(ValueError, match=str(MAX_GRID_BINS)):
        fit_picpi(p, y, num_bins=MAX_GRID_BINS + 1)


def test_fit_rejects_bad_outcome_range() -> None:
    p, y = np.array([0.2, 0.8]), np.array([0.3, 0.7])
    with pytest.raises(ValueError, match="lo < hi"):
        fit_picpi(p, y, outcome_range=(1.0, 0.0))
    with pytest.raises(ValueError, match="lo < hi"):
        fit_picpi(p, y, outcome_range=(0.5, 0.5))
    with pytest.raises(ValueError, match="finite"):
        fit_picpi(p, y, outcome_range=(0.0, float("inf")))


# ---------------------------------------------------------------------------
# Algorithm 1 — population mode
# ---------------------------------------------------------------------------


def test_population_admission_respects_hoeffding_margin() -> None:
    """Line 6 screen recomputed from stored fields: mean in [a+m, b-m]."""
    p_c, y_c, _, _ = synthetic_picpi_data(4_000, 10, 3)
    k, delta = 20, 0.1
    fit = fit_picpi(p_c, y_c, num_bins=k, delta=delta)
    assert fit.lower.size > 0
    expected_margin = 2.0 * np.sqrt(np.log(k * k / delta) / fit.counts)
    np.testing.assert_allclose(fit.margins, expected_margin, rtol=1e-12)
    assert bool((fit.emp_means >= fit.lower + fit.margins - 1e-12).all())
    assert bool((fit.emp_means <= fit.upper - fit.margins + 1e-12).all())
    assert bool((fit.margins <= (fit.upper - fit.lower) / 2.0 + 1e-12).all())
    assert bool((fit.counts > 0).all())
    # sorted by (lower, upper)
    key = np.lexsort((fit.upper, fit.lower))
    np.testing.assert_array_equal(fit.lower, fit.lower[key])


def test_relax_endpoints_admits_boundary_intervals() -> None:
    """Eq. (21): margin skipped only at a=0 / b=1 when the mean cannot exit."""
    p_c, _, _, _ = synthetic_picpi_data(5_000, 10, 5)
    y_zero = np.zeros_like(p_c)
    strict = fit_picpi(p_c, y_zero, num_bins=10, delta=0.1)
    relaxed = fit_picpi(p_c, y_zero, num_bins=10, delta=0.1, relax_endpoints=True)
    assert strict.lower.size == 0  # mean 0 can never clear a + m with m > 0
    assert relaxed.lower.size > 0
    # every relaxed admission must touch a hard boundary (mean 0 => a = 0)
    assert bool((relaxed.lower == 0.0).all())
    assert bool((relaxed.upper > 0.0).all())


def test_regression_outcome_range_rescaling() -> None:
    """Remark 3.6: bounded regression outcomes via affine rescaling."""
    rng = np.random.default_rng(17)
    p = rng.uniform(0.0, 3.0, size=8_000)
    y = p + rng.normal(0.0, 0.3, size=8_000)  # stays inside (-2, 5) w.h.p.
    assert y.min() > -2.0 and y.max() < 5.0
    fit = fit_picpi(p, y, num_bins=20, delta=0.1, outcome_range=(-2.0, 5.0))
    assert fit.lower.size > 0
    assert float(fit.lower.min()) >= -2.0
    assert float(fit.upper.max()) <= 5.0
    # margin condition in rescaled units (span 7)
    span = 7.0
    m_unit = fit.margins / span
    mean_unit = (fit.emp_means - (-2.0)) / span
    a_unit = (fit.lower - (-2.0)) / span
    b_unit = (fit.upper - (-2.0)) / span
    assert bool((mean_unit >= a_unit + m_unit - 1e-12).all())
    assert bool((mean_unit <= b_unit - m_unit + 1e-12).all())
    q = fit.query(1.5)
    assert q.certified or (q.lower == -2.0 and q.upper == 5.0)


def test_query_returns_shortest_containing_interval() -> None:
    p_c, y_c, p_t, _ = synthetic_picpi_data(20_000, 500, 7)
    fit = fit_picpi(p_c, y_c, num_bins=20, delta=0.1)
    for z in (0.13, 0.37, 0.5, 0.71, 0.94):
        q = fit.query(z)
        containing = (fit.lower <= z) & (fit.upper >= z)
        assert q.certified == bool(containing.any())
        if q.certified:
            assert q.width == pytest.approx(float((fit.upper - fit.lower)[containing].min()))
    los, his, cert = fit.query_many(p_t)
    singles = [fit.query(float(z)) for z in p_t[:50]]
    np.testing.assert_allclose(los[:50], [s.lower for s in singles])
    np.testing.assert_allclose(his[:50], [s.upper for s in singles])
    assert bool((cert[:50] == np.array([s.certified for s in singles])).all())


def test_query_fallback_is_trivial_interval_when_nothing_certified() -> None:
    """Section 4.1 fallback: [l(z), u(z)] = [y_lo, y_hi], uncertified."""
    p = np.full(30, 0.5)
    y = np.zeros(30)
    fit = fit_picpi(p, y, num_bins=10, delta=0.1)
    assert fit.lower.size == 0
    q = fit.query(0.5)
    assert not q.certified
    assert (q.lower, q.upper) == (0.0, 1.0)
    assert fit.certified_fraction(np.array([0.1, 0.9])) == 0.0


def test_query_rejects_out_of_range_and_nonfinite() -> None:
    p_c, y_c, _, _ = synthetic_picpi_data(500, 10, 2)
    fit = fit_picpi(p_c, y_c, num_bins=5, delta=0.1)
    with pytest.raises(ValueError, match="finite"):
        fit.query(float("nan"))
    with pytest.raises(ValueError, match="outcome_range"):
        fit.query(1.5)
    with pytest.raises(ValueError, match="outcome_range"):
        fit.query_many(np.array([0.5, -0.2]))
    with pytest.raises(ValueError, match="non-empty"):
        fit.query_many(np.zeros(0))


# ---------------------------------------------------------------------------
# Algorithm 3 — empirical mode
# ---------------------------------------------------------------------------


def test_empirical_partition_is_self_consistent_on_calibration() -> None:
    p_c, y_c, _, _ = synthetic_picpi_data(4_000, 10, 9)
    fit = fit_picpi_empirical(p_c, y_c, num_bins=20)
    assert fit.mode == "empirical"
    assert fit.uncertified_from is None
    assert fit.lower.size > 1
    assert fit.lower[0] == 0.0
    assert fit.upper[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(fit.lower[1:], fit.upper[:-1], atol=1e-12)
    assert bool((fit.counts > 0).all())
    # Line 5: empirical mean strictly inside (t_{j-1}, t_j]
    assert bool((fit.emp_means > fit.lower + 1e-12).all())
    assert bool((fit.emp_means <= fit.upper + 1e-12).all())
    assert bool((fit.margins == 0.0).all())


def test_empirical_stall_leaves_tail_uncertified_failclosed() -> None:
    """Mean-0 outcomes can never satisfy the strict bin condition => stall."""
    p = np.full(50, 0.9)
    y = np.zeros(50)
    fit = fit_picpi_empirical(p, y, num_bins=10)
    assert fit.lower.size == 0
    assert fit.uncertified_from == 0.0
    q = fit.query(0.9)
    assert not q.certified
    assert (q.lower, q.upper) == (0.0, 1.0)


def test_empirical_query_edges() -> None:
    p_c, y_c, _, _ = synthetic_picpi_data(2_000, 10, 4)
    fit = fit_picpi_empirical(p_c, y_c, num_bins=10)
    assert fit.uncertified_from is None
    # half-open bins (t_{j-1}, t_j]: z == 0 is excluded => documented fallback
    q0 = fit.query(0.0)
    assert not q0.certified
    # z == 1 is inside the last bin
    q1 = fit.query(1.0)
    assert q1.certified
    assert q1.upper == pytest.approx(1.0)
    # unique bin assignment: query_many agrees with query
    zs = np.linspace(0.01, 0.99, 25)
    los, his, cert = fit.query_many(zs)
    for i, z in enumerate(zs):
        q = fit.query(float(z))
        assert (los[i], his[i], bool(cert[i])) == (q.lower, q.upper, q.certified)


def test_determinism_both_modes() -> None:
    p_c, y_c, p_t, _ = synthetic_picpi_data(3_000, 200, 21)
    f1 = fit_picpi(p_c, y_c, num_bins=25, delta=0.15)
    f2 = fit_picpi(p_c, y_c, num_bins=25, delta=0.15)
    for a, b in (
        (f1.lower, f2.lower),
        (f1.upper, f2.upper),
        (f1.counts, f2.counts),
        (f1.emp_means, f2.emp_means),
        (f1.margins, f2.margins),
    ):
        np.testing.assert_array_equal(a, b)
    e1 = fit_picpi_empirical(p_c, y_c, num_bins=25)
    e2 = fit_picpi_empirical(p_c, y_c, num_bins=25)
    np.testing.assert_array_equal(e1.lower, e2.lower)
    np.testing.assert_array_equal(e1.upper, e2.upper)
    q1 = fit_picpi(p_c, y_c, num_bins=25, delta=0.15).query_many(p_t)
    q2 = f2.query_many(p_t)
    for a, b in zip(q1, q2, strict=True):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Algorithm 4 — DisjointCalibration
# ---------------------------------------------------------------------------


def test_disjoint_calibration_max_coverage_selection() -> None:
    lo = np.array([0.0, 0.4, 0.55])
    hi = np.array([0.5, 0.6, 1.0])
    dl, du = disjoint_calibration(lo, hi, 1.0)
    np.testing.assert_allclose(dl, [0.0, 0.55])
    np.testing.assert_allclose(du, [0.5, 1.0])


def test_disjoint_calibration_length_filter_and_errors() -> None:
    lo = np.array([0.0, 0.4, 0.55])
    hi = np.array([0.5, 0.6, 1.0])
    dl, du = disjoint_calibration(lo, hi, 0.25)
    np.testing.assert_allclose(dl, [0.4])
    np.testing.assert_allclose(du, [0.6])
    dl, du = disjoint_calibration(lo, hi, 0.1)
    assert dl.size == 0 and du.size == 0
    with pytest.raises(ValueError, match="same length"):
        disjoint_calibration(lo, hi[:2], 1.0)
    with pytest.raises(ValueError, match=">="):
        disjoint_calibration(np.array([0.5]), np.array([0.4]), 1.0)
    with pytest.raises(ValueError, match="max_length"):
        disjoint_calibration(lo, hi, -0.5)
    with pytest.raises(ValueError, match="finite"):
        disjoint_calibration(np.array([np.nan]), np.array([0.4]), 1.0)


def test_disjoint_calibration_on_fit_output_is_sorted_nonoverlapping() -> None:
    p_c, y_c, _, _ = synthetic_picpi_data(20_000, 10, 7)
    fit = fit_picpi(p_c, y_c, num_bins=20, delta=0.1)
    dl, du = disjoint_calibration(fit.lower, fit.upper, 0.4)
    assert dl.size > 0
    assert bool((du >= dl).all())
    assert bool((dl[1:] >= du[:-1] - 1e-12).all())
    assert bool((du - dl <= 0.4 + 1e-12).all())


# ---------------------------------------------------------------------------
# naive contrast + held-out self-consistency
# ---------------------------------------------------------------------------


def test_naive_acceptance_has_no_margin_and_population_mode_rejects() -> None:
    """40 identical predictions with mean 0.30: naive admits [0.25, 0.30];
    Algorithm 1's Hoeffding margin (m ~ 0.91 at N=40) rejects everything."""
    p = np.full(40, 0.27)
    y = np.zeros(40)
    y[:12] = 1.0  # empirical mean exactly 0.30
    nl, nu = naive_bin_acceptance(p, y, num_bins=20)
    hit = (nl <= 0.27) & (nu >= 0.27)
    assert bool(hit.any())
    fit = fit_picpi(p, y, num_bins=20, delta=0.1)
    assert fit.lower.size == 0


def test_adversarial_spike_naive_violates_picpi_holds() -> None:
    """Seeded adversarial prediction distribution (miscalibrated atom at
    0.28, rate 0.45, weight 1.5%): an ordinary ~1-sigma calibration draw
    fools the margin-free naive bin, whose held-out conditional mean then
    exits the bin. Population-mode PICPI rejects the narrow bin structurally
    (margin >> width at K=20) and shows zero held-out violations."""
    seed = 5  # pinned: seed hunt over 0..300, first clean contrast
    p_c, y_c, p_t, y_t = synthetic_picpi_data(
        20_000, 20_000, seed, spike_weight=0.015, spike_value=0.28, spike_rate=0.45
    )
    nl, nu = naive_bin_acceptance(p_c, y_c, num_bins=20)
    hit = (nl <= 0.28) & (nu >= 0.28)
    assert bool(hit.any()), "naive must accept the spike bin at this seed"
    a, b = float(nl[hit][0]), float(nu[hit][0])
    mask = (p_t >= a) & (p_t <= b)
    held_mean = float(y_t[mask].mean())
    assert not (a <= held_mean <= b), "naive bin must be violated on held-out data"
    assert held_mean > b

    fit = fit_picpi(p_c, y_c, num_bins=20, delta=0.1)
    # the narrow spike bin is never admitted
    narrow = (fit.upper - fit.lower) <= (b - a) + 1e-12
    contains = (fit.lower[narrow] <= 0.28) & (fit.upper[narrow] >= 0.28)
    assert not bool(contains.any())
    report = holdout_self_consistency(fit.lower, fit.upper, p_t, y_t, min_count=200)
    assert report["n_checked"] > 20
    assert report["violation_rate"] == 0.0
    naive_report = holdout_self_consistency(nl, nu, p_t, y_t, min_count=200)
    assert naive_report["violation_rate"] > 0.0
    # query at the spike: either uncertified fallback, or a certified
    # interval whose held-out conditional mean stays inside it
    q = fit.query(0.28)
    if q.certified:
        in_q = (p_t >= q.lower) & (p_t <= q.upper)
        assert in_q.sum() >= 200
        assert q.lower <= float(y_t[in_q].mean()) <= q.upper


def test_holdout_self_consistency_calibrated_dgp() -> None:
    """On a calibrated SYNTHETIC DGP every admitted interval is a genuine
    population PICPI; held-out empirical means stay inside (noise-gated)."""
    p_c, y_c, p_t, y_t = synthetic_picpi_data(40_000, 40_000, 13)
    fit = fit_picpi(p_c, y_c, num_bins=50, delta=0.1)
    report = holdout_self_consistency(fit.lower, fit.upper, p_t, y_t, min_count=200)
    assert report["n_checked"] >= 20
    assert report["violation_rate"] <= 0.05
    assert fit.certified_fraction(p_t) >= 0.95
    with pytest.raises(ValueError, match="min_count"):
        holdout_self_consistency(fit.lower, fit.upper, p_t, y_t, min_count=0)
    with pytest.raises(ValueError, match="same length"):
        holdout_self_consistency(fit.lower, fit.upper[:-1], p_t, y_t)


def test_coverage_profile_monotone_and_matches_certified_fraction() -> None:
    """Eq. (6) diagnostic: covered mass is monotone in the width cap."""
    p_c, y_c, p_t, _ = synthetic_picpi_data(20_000, 5_000, 7)
    fit = fit_picpi(p_c, y_c, num_bins=50, delta=0.1)
    caps = [0.02, 0.1, 0.2, 0.4, 0.8, 1.0]
    prof = fit.coverage_profile(p_t, caps)
    assert bool((np.diff(prof) >= -1e-12).all())
    assert prof[0] == 0.0  # nothing admitted below the theoretical width floor
    assert prof[-1] >= 0.9
    assert prof[-1] == pytest.approx(fit.certified_fraction(p_t))
    with pytest.raises(ValueError, match="non-negative"):
        fit.coverage_profile(p_t, [-1.0])
    with pytest.raises(ValueError, match="non-empty"):
        fit.coverage_profile(p_t, [])


def test_certified_fraction_monotone_in_delta() -> None:
    """Larger delta => smaller margin => weakly more certified mass."""
    p_c, y_c, p_t, _ = synthetic_picpi_data(20_000, 5_000, 7)
    cf_loose = fit_picpi(p_c, y_c, num_bins=20, delta=0.3).certified_fraction(p_t)
    cf_tight = fit_picpi(p_c, y_c, num_bins=20, delta=0.05).certified_fraction(p_t)
    assert cf_loose >= cf_tight
    assert cf_loose >= 0.9


def test_expand_epsilon_clips_to_range() -> None:
    """Eq. (13): epsilon-expansion of the queried interval, clipped."""
    p_c, y_c, _, _ = synthetic_picpi_data(20_000, 10, 7)
    fit = fit_picpi(p_c, y_c, num_bins=20, delta=0.1)
    zs = np.array([0.02, 0.5, 0.98])
    los, his, _ = fit.query_many(zs)
    elo, ehi = fit.expand(zs, 0.05)
    np.testing.assert_allclose(elo, np.maximum(los - 0.05, 0.0))
    np.testing.assert_allclose(ehi, np.minimum(his + 0.05, 1.0))
    with pytest.raises(ValueError, match="epsilon"):
        fit.expand(zs, -0.1)


# ---------------------------------------------------------------------------
# Theorem 3.2 width-rate illustration (SYNTHETIC, not a proof)
# ---------------------------------------------------------------------------


def test_width_rate_slope_synthetic() -> None:
    """log-log slope of mean minimal certified width vs n ~ -1/3.

    Illustration of Theorem 3.2's rate on the calibrated fixture (eps = 0,
    lambda = 1), following the Appendix B.5 protocol (width 1.0 when no
    admitted interval contains the prediction). Generous tolerance: the
    fixed K-grid floors widths at 1/K, flattening the observed slope.
    """
    out = bench_picpi_width_rate(
        (10_000, 40_000, 160_000, 640_000), num_bins=100, delta=0.1, n_test=2_000, seed=11
    )
    slope = float(out["width_slope"])
    assert -0.55 <= slope <= -0.18, f"expected ~n^(-1/3), got slope {slope:.3f}"
    assert float(out["width_last"]) < float(out["width_first"])
    assert float(out["certified_fraction_last"]) >= 0.9
    assert out["claim"] == "research_metric_only"
    assert out["dgp"] == "fixture"


def test_width_rate_validation() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        bench_picpi_width_rate((1_000, 2_000))
    with pytest.raises(ValueError, match=">= 100"):
        bench_picpi_width_rate((10, 200, 300))
    with pytest.raises(ValueError, match="delta"):
        bench_picpi_width_rate((1_000, 2_000, 3_000), delta=1.5)


# ---------------------------------------------------------------------------
# Section 4.2 — multiclass label sets
# ---------------------------------------------------------------------------


def test_multiclass_fit_validation() -> None:
    q, lab, _, _ = _oracle_multiclass(200, 50, 1, n_classes=3)
    absent = lab.copy()
    absent[absent == 2] = 0  # class 2 disappears
    with pytest.raises(ValueError, match="absent"):
        fit_picpi_multiclass(q, absent, num_bins=5)
    with pytest.raises(ValueError, match="integers"):
        fit_picpi_multiclass(q, lab.astype(float), num_bins=5)
    with pytest.raises(ValueError, match=r"\[0, G-1\]"):
        fit_picpi_multiclass(q, lab + 3, num_bins=5)
    with pytest.raises(ValueError, match="align"):
        fit_picpi_multiclass(q, lab[:-1], num_bins=5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_picpi_multiclass(q * 1.5, lab, num_bins=5)
    with pytest.raises(ValueError, match="2-d"):
        fit_picpi_multiclass(q.ravel(), lab, num_bins=5)
    with pytest.raises(ValueError, match="delta"):
        fit_picpi_multiclass(q, lab, num_bins=5, delta_m=0.0)
    with pytest.raises(ValueError, match="max_interval_length"):
        fit_picpi_multiclass(q, lab, num_bins=5, max_interval_length=0.0)


def test_multiclass_structure_and_certificate() -> None:
    q_cal, lab_cal, q_test, lab_test = _oracle_multiclass(40_000, 12_000, 23)
    mc = fit_picpi_multiclass(q_cal, lab_cal, num_bins=40, delta=0.3, delta_m=0.05, delta_pi=0.05)
    assert mc.n_classes == 3
    for g in range(3):
        lo_g, up_g = mc.lower[g], mc.upper[g]
        if lo_g.size > 1:
            assert bool((lo_g[1:] >= up_g[:-1] - 1e-12).all())  # disjoint, sorted
        assert bool((up_g >= lo_g).all())
        assert bool((mc.masses[g] >= 0.0).all()) and bool((mc.masses[g] <= 1.0).all())
        assert float(mc.masses[g].sum()) <= 1.0 + 1e-9
        assert mc.priors[g] == pytest.approx(float(np.mean(lab_cal == g)))
    assert mc.eps_m > 0.0 and mc.eps_pi > 0.0

    alphas = (0.35, 0.35, 0.35)
    rule = mc.select(alphas)
    assert any(rule.feasible), "at least one class must admit a feasible subset"
    for g in range(3):
        gamma = mc.gamma_hat(g, rule.keep[g])[2]
        assert rule.gamma_hat[g] == pytest.approx(gamma)
        if rule.feasible[g]:
            assert gamma <= alphas[g] + 1e-9
        else:
            assert bool(rule.keep[g].all())  # fail-closed: keep everything
    pred = mc.predict(rule, q_test)
    sizes = pred.sum(axis=1)
    assert float(sizes.mean()) <= 3.0
    for g in range(3):
        rows = lab_test == g
        assert int(rows.sum()) > 500
        miscoverage = float(np.mean(~pred[rows, g]))
        assert miscoverage <= alphas[g] + 0.10, (g, miscoverage)


def test_multiclass_tight_alpha_is_failclosed() -> None:
    q_cal, lab_cal, _, _ = _oracle_multiclass(6_000, 100, 31)
    mc = fit_picpi_multiclass(q_cal, lab_cal, num_bins=20, delta=0.3)
    rule = mc.select((0.0, 0.0, 0.0))
    for g in range(3):
        if not rule.feasible[g]:
            assert bool(rule.keep[g].all())
            # the keep-everything certificate is still reported honestly
            assert rule.gamma_hat[g] == pytest.approx(mc.gamma_hat(g, rule.keep[g])[2])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        mc.select((0.35, 1.0, 0.35))
    with pytest.raises(ValueError, match="one entry per class"):
        mc.select((0.35, 0.35))


def test_gamma_hat_matches_hand_computation() -> None:
    """Theorem 4.5 radii on a hand-built single-class table."""
    mc = MulticlassPicpi(
        lower=(np.array([0.2]),),
        upper=(np.array([0.8]),),
        masses=(np.array([0.9]),),
        priors=(0.5,),
        eps_m=0.01,
        eps_pi=0.01,
        n_cal=1_000,
        n_classes=1,
        num_bins=10,
        delta=0.1,
        delta_m=0.05,
        delta_pi=0.05,
    )
    up_all, low_all, gam_all = mc.gamma_hat(0, np.array([True]))
    # m_low = 0.89, pi_low = 0.49, pi_high = 0.51
    assert up_all == pytest.approx((1.0 - 0.89) / 0.49)
    assert low_all == pytest.approx(1.0 - 0.2 * 0.89 / 0.51)
    assert gam_all == pytest.approx(min(up_all, low_all))
    up_none, low_none, gam_none = mc.gamma_hat(0, np.array([False]))
    assert up_none == pytest.approx((1.0 - (1.0 - 0.8) * 0.89) / 0.49)
    assert low_none == pytest.approx(1.0)
    assert gam_none == pytest.approx(min(up_none, low_none))
    with pytest.raises(ValueError, match="class index"):
        mc.gamma_hat(1, np.array([True]))
    with pytest.raises(ValueError, match="interval count"):
        mc.gamma_hat(0, np.array([True, False]))


def test_min_weight_cover_matches_brute_force() -> None:
    rng = np.random.default_rng(5)
    for _trial in range(12):
        j = int(rng.integers(2, 8))
        profits = np.round(rng.uniform(0.05, 1.0, size=j), 3)
        weights = np.round(rng.uniform(0.1, 2.0, size=j), 3)
        target = float(rng.uniform(0.1, profits.sum()))
        mask = _min_weight_cover(profits, weights, target)
        best_w, best_mask = float("inf"), None
        for bits in itertools.product((False, True), repeat=j):
            sel = np.asarray(bits, dtype=bool)
            if float(profits[sel].sum()) >= target - 1e-9:
                w = float(weights[sel].sum())
                if w < best_w:
                    best_w, best_mask = w, sel
        if best_mask is None:
            assert mask is None
        else:
            assert mask is not None
            assert float(profits[mask].sum()) >= target - 1e-6
            assert float(weights[mask].sum()) <= best_w + 1e-6
    # edges
    zero = _min_weight_cover(np.array([0.5]), np.array([1.0]), 0.0)
    assert zero is not None and not bool(zero.any())
    assert _min_weight_cover(np.array([0.4]), np.array([1.0]), 0.5) is None
    assert _min_weight_cover(np.zeros(0), np.zeros(0), 0.5) is None
    with pytest.raises(ValueError, match="non-negative"):
        _min_weight_cover(np.array([-0.1]), np.array([1.0]), 0.05)


# ---------------------------------------------------------------------------
# SYNTHETIC fixture + bench hygiene
# ---------------------------------------------------------------------------


def test_synthetic_data_determinism_and_spike() -> None:
    a = synthetic_picpi_data(1_000, 500, 42, spike_weight=0.05)
    b = synthetic_picpi_data(1_000, 500, 42, spike_weight=0.05)
    for x, y_ in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y_)
    p_c, y_c, _, _ = a
    assert bool((p_c == 0.28).any())
    assert bool(((y_c == 0.0) | (y_c == 1.0)).all())
    with pytest.raises(ValueError, match="spike_weight"):
        synthetic_picpi_data(100, 100, 1, spike_weight=1.0)
    with pytest.raises(ValueError, match=">= 1"):
        synthetic_picpi_data(0, 10, 1)


def test_bench_picpi_fixture_keys_and_honesty() -> None:
    out = bench_picpi(n_cal=4_000, n_test=4_000, num_bins=20, delta=0.1, seed=7)
    assert out["claim"] == "research_metric_only"
    assert out["dgp"] == "fixture"
    assert out["seed"] == 7.0
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav")
    for key in out:
        assert not any(tok in key.lower() for tok in forbidden)
    assert 0.0 <= float(out["certified_fraction"]) <= 1.0
    assert float(out["selfconsistency_violation_rate"]) >= 0.0
    assert float(out["n_intervals"]) >= 0.0
    assert np.isfinite(float(out["mean_width"]))


def test_bench_picpi_panel_passthrough_and_errors() -> None:
    p_c, y_c, p_t, y_t = synthetic_picpi_data(2_000, 2_000, 3)
    out = bench_picpi(p_cal=p_c, y_cal=y_c, p_test=p_t, y_test=y_t, num_bins=20)
    assert out["dgp"] == "panel"
    assert "seed" not in out
    with pytest.raises(ValueError, match="all panel"):
        bench_picpi(p_cal=p_c)
