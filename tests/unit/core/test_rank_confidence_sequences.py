"""Tests for rank_confidence_sequences: anytime-valid rank inference (wave-13 A4).

Constructions under test:
- Rank confidence sequences (Khosravi & Huo 2026, arXiv:2609.32211):
  pairwise betting e-processes (their Eq. (2)-(3)), closed testing over weak
  orders by enumeration (Algorithm 1, exact, M <= 8), the polynomial-time
  transitivity-pooled shortcut (Algorithm 2), rank sets / intervals / tiers /
  top-k reports (Theorems 4.2-4.4);
- BB-EDGE (Gao, Zhang, Xie, Jing, Wang & Liu 2026, arXiv:2609.32248):
  benchmark-weighted block-factorized empirical-Bernstein e-processes
  (Eqs. (3)-(7), Proposition 1), direct e-Holm edges with anytime FWER
  control (Eq. (8), Theorem 1; Hartog & Lei 2025 arXiv:2501.09015), Top-k
  certification (Appendix A, Theorem 2) and simultaneous rank intervals
  (Appendix B, Corollary 1).

All data are seeded SYNTHETIC item-score matrices (np.random.default_rng,
pinned seeds — determinism, no market data, no market evidence; rank
inference over model scores is a research diagnostic only). Monte-Carlo
tolerances follow the repo convention (test_confidence_sequences.py /
test_e_detectors.py): with R replicates at level alpha the empirical
time-uniform violation rate / FWER must stay <= alpha + 0.02 (binomial
slack). Within-item and within-replicate dependence is injected via a common
item-difficulty Gaussian factor (arbitrary dependence is allowed by both
constructions). Observed violations in the seeded runs below are 0; the
tolerance guards MC noise only.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.rank_confidence_sequences import (
    _RCS_MAX_K_EXACT,
    BBEdgeResult,
    RankConfidenceSequenceResult,
    bb_edge_certify,
    bb_edge_e_values,
    bb_edge_topk_certify,
    check_dominance_fwer,
    check_edge_fwer,
    check_rank_set_coverage,
    check_rank_time_uniform_coverage,
    closed_testing_rank_sets,
    direct_e_holm,
    direction_pairs,
    enumerate_weak_orders,
    pairwise_betting_wealth,
    rank_confidence_sequence,
    shortcut_certify,
    topk_status,
)

Array = NDArray[np.float64]

SEED = 20260929
ALPHA = 0.05
# Documented MC tolerance: binomial slack at ~20-100 replicates (repo
# convention). The seeded runs observe violation/FWER counts of 0; the slack
# only absorbs regeneration noise, never a systematic breach.
VIOL_TOL = 0.02
FUBINI = {2: 3, 3: 13, 4: 75, 5: 541, 6: 4683, 7: 47293, 8: 545835}


def _copula_binary_panels(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    theta: Array,
    kappa: float,
) -> Array:
    """SYNTHETIC binary scores with a common item-difficulty factor.

    Z = sqrt(kappa) * U_item + sqrt(1 - kappa) * eps; X = 1{Z <= Phi^-1(theta_j)}.
    E[X_tj] = theta_j exactly, and kappa > 0 makes the within-item scores
    dependent across models in an arbitrary (Gaussian-copula) way. ``shape``
    is (n_reps, T) or (R,) broadcast over items; theta has one entry per
    model (last axis of the output).
    """
    m = int(np.asarray(theta).size)
    lead = shape + (1,)
    u_item = rng.standard_normal(lead)
    eps = rng.standard_normal(shape + (m,))
    z = np.sqrt(kappa) * u_item + np.sqrt(1.0 - kappa) * eps
    thr = norm.ppf(np.asarray(theta, dtype=float))
    return (z <= thr).astype(float)


def _true_ranks(theta: Array) -> NDArray[np.int64]:
    """R_j = 1 + #{l: theta_l > theta_j} (ties share the better rank)."""
    th = np.asarray(theta, dtype=float).reshape(-1)
    return np.asarray(1 + (th[:, None] < th[None, :]).sum(axis=1), dtype=np.int64)


# ---------------------------------------------------------------------------
# weak orders (closed-testing atoms)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", sorted(FUBINI))
def test_weak_order_counts_match_fubini(k: int) -> None:
    """Enumeration yields exactly the Fubini (ordered Bell) numbers."""
    levels = enumerate_weak_orders(k)
    assert levels.shape == (FUBINI[k], k)


def test_weak_orders_canonical_unique_and_deterministic() -> None:
    """Levels are contiguous from 0, rows unique, regeneration identical."""
    levels = enumerate_weak_orders(5)
    for row in levels:
        assert set(row.tolist()) == set(range(int(row.max()) + 1))
    uniq = np.unique(levels, axis=0)
    assert uniq.shape[0] == levels.shape[0]
    assert np.array_equal(enumerate_weak_orders(5), levels)  # cache/determinism


def test_direction_pairs_layout() -> None:
    pairs = direction_pairs(4)
    assert len(pairs) == 12
    assert pairs[0] == (0, 1)
    assert all(a != b for a, b in pairs)
    with pytest.raises(ValueError, match="at least 2"):
        direction_pairs(1)


# ---------------------------------------------------------------------------
# Step 1: pairwise betting wealths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["superpopulation", "finite"])
def test_pairwise_wealth_tied_scores_stay_at_one(sampling: str) -> None:
    """Z_t == 0 => every factor is 1 => wealth identically 1 (both models)."""
    scores = np.full((30, 3), 0.4)
    w = pairwise_betting_wealth(scores, sampling=sampling)
    assert w.shape == (31, 3, 3)
    assert np.all(w == 1.0)


def test_pairwise_wealth_dominant_pair_grows_and_reverse_decays() -> None:
    """Model 0 always outscores model 1: E^{01} grows, E^{10} <= 1."""
    scores = np.zeros((30, 2))
    scores[:, 0] = 1.0
    w = pairwise_betting_wealth(scores, sampling="superpopulation")
    e01 = w[:, 0, 1]
    e10 = w[:, 1, 0]
    assert np.all(np.diff(e01) > 0.0)  # Z = +1 => factors 1 + lambda > 1
    assert np.all(e10[1:] < 1.0) and np.all(e10 > 0.0)
    assert np.all(np.isfinite(w))


def test_pairwise_wealth_finite_mode_offset_bounded() -> None:
    """Under (F) the offset stays in [-0.99, 1] and wealths stay nonnegative."""
    rng = np.random.default_rng(SEED)
    scores = rng.random((50, 3))
    w = pairwise_betting_wealth(scores, sampling="finite")
    assert np.all(w >= 0.0) and np.all(np.isfinite(w))
    assert np.all(w[0] == 1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sampling": "nonsense"}, "sampling"),
        ({"grid": (1.0,)}, r"\[0, 1\)"),
        ({"grid": ()}, "nonempty"),
        ({"grid": (-0.1, 0.2)}, r"\[0, 1\)"),
        ({"e_cap": 0.5}, "e_cap"),
    ],
)
def test_pairwise_wealth_fail_closed(kwargs: dict, match: str) -> None:
    scores = np.full((10, 2), 0.5)
    with pytest.raises(ValueError, match=match):
        pairwise_betting_wealth(scores, **kwargs)  # type: ignore[arg-type]


def test_score_matrix_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pairwise_betting_wealth(np.array([[1.5, 0.2], [0.1, 0.2]]))
    with pytest.raises(ValueError, match="finite"):
        pairwise_betting_wealth(np.array([[np.nan, 0.2], [0.1, 0.2]]))
    with pytest.raises(ValueError, match="n_models>=2"):
        pairwise_betting_wealth(np.ones((10, 1)))


# ---------------------------------------------------------------------------
# Steps 2-3: closed testing (exact) and shortcut
# ---------------------------------------------------------------------------


def test_exact_t0_report_is_trivially_widest() -> None:
    rng = np.random.default_rng(SEED + 1)
    res = rank_confidence_sequence(rng.random((20, 4)), alpha=ALPHA, method="exact")
    assert not res.dominance[0].any()
    assert np.all(res.rank_lower[0] == 1) and np.all(res.rank_upper[0] == 4)
    assert res.rank_sets is not None
    assert np.all(res.rank_sets[0])  # every rank still possible at t = 0
    assert np.all(res.tiers[0] == 1)


def test_exact_dominant_model_recovers_order() -> None:
    """Model 0 scores 1, the rest tie at 0: 0 > l certified, others unresolved."""
    scores = np.zeros((60, 4))
    scores[:, 0] = 1.0
    res = rank_confidence_sequence(scores, alpha=ALPHA, method="exact")
    dom = res.dominance[-1]
    assert np.all(dom[0, 1:]) and not dom[1:, 0].any()
    assert not dom[1:, 1:].any()  # tied models: nothing certified between them
    assert res.rank_lower[-1, 0] == 1 and res.rank_upper[-1, 0] == 1
    assert np.all(res.rank_lower[-1, 1:] == 2) and np.all(res.rank_upper[-1, 1:] == 4)
    assert res.rank_sets is not None
    assert res.rank_sets[-1, 0].tolist() == [True, False, False, False]
    assert res.tiers[-1, 0] == 1 and np.all(res.tiers[-1, 1:] == 2)
    assert not res.inconsistent.any()
    tk = res.top_k(1)
    assert tk["inside"][-1].tolist() == [True, False, False, False]
    assert tk["outside"][-1].tolist() == [False, True, True, True]


def test_exact_method_capped_at_documented_k() -> None:
    """M > _RCS_MAX_K_EXACT fails closed for 'exact'; 'shortcut' supports it."""
    assert _RCS_MAX_K_EXACT == 8
    rng = np.random.default_rng(SEED + 2)
    scores = rng.random((10, 9))
    with pytest.raises(ValueError, match="capped"):
        rank_confidence_sequence(scores, alpha=ALPHA, method="exact")
    res = rank_confidence_sequence(scores, alpha=ALPHA, method="shortcut")
    assert res.rank_sets is None and res.dominance.shape == (11, 9, 9)


def test_shortcut_is_transitively_closed() -> None:
    """Closure step (Algorithm 2, step 7): chains certified across times."""
    m = 4
    wealth = np.ones((3, m, m))
    wealth[1, 0, 1] = 250.0  # (0,1) certifiable at t=1 via its own wealth
    wealth[2, 0, 1] = 1.0  # wealth drops back at t=2 ...
    wealth[2, 1, 2] = 250.0  # ... while (1,2) certifies at t=2
    res = shortcut_certify(wealth, alpha=ALPHA)
    assert res.dominance[1, 0, 1] and not res.dominance[1, 1, 2]
    assert res.dominance[2, 0, 1] and res.dominance[2, 1, 2]  # permanence
    assert res.dominance[2, 0, 2]  # added ONLY by the transitive closure
    # model 2 has two certified superiors (0 via closure, 1 directly); model 0
    # dominates {1, 2} but not 3, so its interval is [1, 4-2] = [1, 2].
    assert res.rank_lower[-1, 2] == 3 and res.rank_upper[-1, 0] == 2
    assert not res.inconsistent.any()


def test_shortcut_subset_of_exact_and_contains_e_bonferroni() -> None:
    """Paper's ordering: e-Bonferroni subset shortcut subset exact (Thm B.8)."""
    rng = np.random.default_rng(SEED + 3)
    theta = np.array([0.7, 0.55, 0.45, 0.3])
    scores = _copula_binary_panels(rng, (40, 1), theta, 0.4)[:, 0, :]
    wealth = pairwise_betting_wealth(scores)
    exact = closed_testing_rank_sets(wealth, alpha=ALPHA)
    short = shortcut_certify(wealth, alpha=ALPHA)
    assert not np.any(short.dominance & ~exact.dominance)  # shortcut subset exact
    thresh = 4 * 3 / ALPHA
    eb = (wealth[1:].max(axis=0) >= thresh) & ~np.eye(4, dtype=bool)
    assert not np.any(eb & ~short.dominance[-1])  # e-Bonferroni subset shortcut


def test_closed_testing_inconsistent_fails_closed() -> None:
    """All orderings eliminated (bad event, prob <= alpha): widest reports."""
    wealth = np.full((3, 3, 3), 1e6)
    wealth[0] = 1.0
    res = closed_testing_rank_sets(wealth, alpha=ALPHA)
    assert res.inconsistent[1] and res.inconsistent[2]
    assert not res.dominance.any()
    assert np.all(res.rank_lower == 1) and np.all(res.rank_upper == 3)
    assert res.rank_sets is not None and not res.rank_sets[1:].any()


def test_rank_confidence_sequence_fail_closed_validation() -> None:
    rng = np.random.default_rng(SEED + 4)
    scores = rng.random((10, 3))
    with pytest.raises(ValueError, match="alpha"):
        rank_confidence_sequence(scores, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        rank_confidence_sequence(scores, alpha=1.0)
    with pytest.raises(ValueError, match="method"):
        rank_confidence_sequence(scores, method="ip")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sampling"):
        rank_confidence_sequence(scores, sampling="bootstrap")
    with pytest.raises(ValueError, match="nonnegative"):
        shortcut_certify(np.full((2, 3, 3), -1.0), alpha=ALPHA)
    with pytest.raises(ValueError, match="wealth-path"):
        shortcut_certify(np.ones((1, 3, 3)), alpha=ALPHA)


def test_topk_status_from_intervals() -> None:
    lower = np.array([1, 2, 2, 4])
    upper = np.array([1, 3, 3, 4])
    tk = topk_status(lower, upper, 1)
    assert tk["inside"].tolist() == [True, False, False, False]
    # lower_j > k: models 1,2 have a certified superior, model 3 has three
    assert tk["outside"].tolist() == [False, True, True, True]
    tk3 = topk_status(lower, upper, 3)
    assert tk3["inside"].tolist() == [True, True, True, False]
    assert tk3["outside"].tolist() == [False, False, False, True]
    with pytest.raises(ValueError, match=r"\[1, M\]"):
        topk_status(lower, upper, 5)


# ---------------------------------------------------------------------------
# BB-EDGE: direct e-Holm, e-processes, edges, top-k
# ---------------------------------------------------------------------------


def test_direct_e_holm_threshold_formula() -> None:
    """c = 1/alpha + sum_{E < 1/alpha}(1/alpha - E) (Gao et al. Eq. (8))."""
    c, rej = direct_e_holm(np.array([100.0, 0.5]), 0.05)
    assert c == pytest.approx(20.0 + (20.0 - 0.5))
    assert rej.tolist() == [True, False]
    c, rej = direct_e_holm(np.array([25.0, 30.0]), 0.05)  # J empty => c = 1/alpha
    assert c == pytest.approx(20.0)
    assert rej.tolist() == [True, True]
    c, rej = direct_e_holm(np.array([1.0, 1.0]), 0.05)
    assert c == pytest.approx(20.0 + 19.0 + 19.0)
    assert not rej.any()
    with pytest.raises(ValueError, match="nonempty"):
        direct_e_holm(np.array([]), 0.05)
    with pytest.raises(ValueError, match="nonnegative"):
        direct_e_holm(np.array([-1.0]), 0.05)


def test_bb_edge_constant_scores_no_edges() -> None:
    """All models identical: Y == mu0, prediction == mu0, E == 1, no edges."""
    panels = np.full((12, 20, 3), 0.5)
    res = bb_edge_certify(panels, tau=0.0, alpha=ALPHA)
    assert np.all(res.e_values == pytest.approx(1.0))
    assert not res.edges.any()
    assert np.all(res.rank_lower == 1) and np.all(res.rank_upper == 3)
    assert not res.inconsistent.any()


def test_bb_edge_weight_proportional_stakes_preserve_heterogeneous_null() -> None:
    """Proposition 1: block means straddle mu0 but the weighted average is 0.

    Two blocks (w = 0.25 / 0.75); direction (0->1) block means are mu0+0.1
    and mu0-1/30, so the benchmark-average null Delta = 0 holds while each
    block individually violates it. Weight-proportional stakes make the total
    linear drift exactly zero: E <= 1 at every replicate, no edge ever.
    """
    n_rep, l0, l1 = 10, 10, 30
    blocks = np.array([0] * l0 + [1] * l1)
    panels = np.empty((n_rep, l0 + l1, 2))
    # block 0: mean difference +0.2 => Y0 = mu0 + 0.1
    panels[:, :l0, 0] = 0.6
    panels[:, :l0, 1] = 0.4
    # block 1: mean difference -1/15 => Y1 = mu0 - 1/30
    panels[:, l0:, 0] = 0.5 - 1.0 / 30.0
    panels[:, l0:, 1] = 0.5 + 1.0 / 30.0
    e = bb_edge_e_values(panels, blocks=blocks, tau=0.0)
    assert np.all(e <= 1.0 + 1e-12)
    res = bb_edge_certify(panels, blocks=blocks, tau=0.0, alpha=ALPHA)
    assert not res.edges.any()


def test_bb_edge_deterministic_dominance_certifies() -> None:
    """Noiseless gap 0.4: true direction certified, reverse never, rank 1."""
    panels = np.full((60, 10, 3), 0.5)
    panels[:, :, 0] = 0.9
    res = bb_edge_certify(panels, tau=0.0, alpha=ALPHA)
    assert res.edges[-1, 0, 1] and res.edges[-1, 0, 2]
    assert not res.edges[:, 1, 0].any() and not res.edges[:, 2, 0].any()
    assert not res.edges[:, 1, 2].any() and not res.edges[:, 2, 1].any()  # tied
    assert res.rank_lower[-1, 0] == 1 and res.rank_upper[-1, 0] == 1
    assert np.all(res.rank_lower[-1, 1:] == 2) and np.all(res.rank_upper[-1, 1:] == 3)
    assert not res.inconsistent.any()


def test_bb_edge_margin_tau_shifts_null() -> None:
    """Gap 0.3 certifies against tau = 0 but not against the wider tau = 0.25.

    Deterministic constant panels: with tau = 0.25 the drift per replicate is
    ~0.024, far too slow to reach the e-Holm threshold within R = 60, while
    with tau = 0 (drift ~0.14/replicate) the edge certifies by r ~= 53.
    """
    panels = np.full((60, 10, 2), 0.4)
    panels[:, :, 0] = 0.7
    res_margin = bb_edge_certify(panels, tau=0.25, alpha=ALPHA)
    assert not res_margin.edges.any()
    res_plain = bb_edge_certify(panels, tau=0.0, alpha=ALPHA)
    assert res_plain.edges[:, 0, 1].any()
    assert not res_plain.edges[:, 1, 0].any()


def test_bb_edge_topk_pilot_deterministic() -> None:
    pilot = np.full((2, 10, 3), 0.5)
    pilot[:, :, 0] = 0.9
    confirm = np.full((60, 10, 3), 0.5)
    confirm[:, :, 0] = 0.9
    tk = bb_edge_topk_certify(pilot, confirm, 1, tau=0.0, alpha=ALPHA)
    assert tk.candidate.tolist() == [0]
    assert tk.certified and tk.certified_at is not None and tk.certified_at < 60
    assert len(tk.directions) == 2  # k(L-k) cross-set directions only
    assert set(tk.directions) == {(0, 1), (0, 2)}


def test_bb_edge_fail_closed_validation() -> None:
    panels = np.full((5, 10, 3), 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bb_edge_certify(np.full((5, 10, 3), 1.2))
    with pytest.raises(ValueError, match="3-D"):
        bb_edge_certify(np.full((10, 3), 0.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tau"):
        bb_edge_certify(panels, tau=1.0)
    with pytest.raises(ValueError, match="blocks"):
        bb_edge_certify(panels, blocks=[0, 1, 2])
    with pytest.raises(ValueError, match="blocks"):
        bb_edge_certify(panels, blocks=[-1] * 10)
    with pytest.raises(ValueError, match="mixture"):
        bb_edge_certify(panels, mixture=[1.0, 2.0])
    with pytest.raises(ValueError, match="directions"):
        bb_edge_certify(panels, directions=[(0, 0)])
    with pytest.raises(ValueError, match="directions"):
        bb_edge_certify(panels, directions=[(0, 9)])
    with pytest.raises(ValueError, match="1 <= k < L"):
        bb_edge_topk_certify(panels, panels, 3)
    with pytest.raises(ValueError, match="1 <= k < L"):
        bb_edge_topk_certify(panels, panels, 0)
    with pytest.raises(ValueError, match="share"):
        bb_edge_topk_certify(panels, np.full((5, 8, 3), 0.5), 1)


def test_determinism_repeated_calls_bitwise_identical() -> None:
    rng = np.random.default_rng(SEED + 5)
    scores = rng.random((30, 4))
    r1 = rank_confidence_sequence(scores, alpha=ALPHA, method="exact")
    r2 = rank_confidence_sequence(scores, alpha=ALPHA, method="exact")
    assert np.array_equal(r1.wealth, r2.wealth)
    assert np.array_equal(r1.dominance, r2.dominance)
    assert r1.rank_sets is not None and r2.rank_sets is not None
    assert np.array_equal(r1.rank_sets, r2.rank_sets)
    panels = rng.random((6, 12, 3))
    b1 = bb_edge_certify(panels, alpha=ALPHA)
    b2 = bb_edge_certify(panels, alpha=ALPHA)
    assert np.array_equal(b1.e_values, b2.e_values)
    assert np.array_equal(b1.edges, b2.edges)


# ---------------------------------------------------------------------------
# Monte-Carlo validation (seeded SYNTHETIC; tolerances documented above)
# ---------------------------------------------------------------------------


def test_rcs_time_uniform_coverage_under_within_item_dependence() -> None:
    """Rank sets AND intervals cover all true ranks at all times (Thm 4.2/4.3).

    Superpopulation model (S): 4 models, Gaussian-copula item factor
    (kappa = 0.5 => strong arbitrary within-item dependence), binary scores,
    T = 80 items, 100 seeded replicates. Also checks coverage at a
    DATA-DEPENDENT stopping time (first all-singleton interval, else T) —
    the anytime-validity claim of the paper.
    """
    rng = np.random.default_rng(SEED + 10)
    theta = np.array([0.62, 0.55, 0.48, 0.41])
    true_ranks = _true_ranks(theta)
    n_reps, t_len = 100, 80
    panels = _copula_binary_panels(rng, (n_reps, t_len), theta, 0.5)
    los, ups, sets, slos, sups = [], [], [], [], []
    for r in range(n_reps):
        exact = rank_confidence_sequence(panels[r], alpha=ALPHA, method="exact")
        short = rank_confidence_sequence(panels[r], alpha=ALPHA, method="shortcut")
        los.append(exact.rank_lower)
        ups.append(exact.rank_upper)
        assert exact.rank_sets is not None
        sets.append(exact.rank_sets)
        slos.append(short.rank_lower)
        sups.append(short.rank_upper)
    cov = check_rank_time_uniform_coverage(np.asarray(los), np.asarray(ups), true_ranks)
    assert float(cov["violation_rate"]) <= ALPHA + VIOL_TOL
    cov_sets = check_rank_set_coverage(np.asarray(sets), true_ranks)
    assert float(cov_sets["violation_rate"]) <= ALPHA + VIOL_TOL
    cov_s = check_rank_time_uniform_coverage(np.asarray(slos), np.asarray(sups), true_ranks)
    assert float(cov_s["violation_rate"]) <= ALPHA + VIOL_TOL
    # data-dependent stopping time: first t where every interval is a singleton
    lo_a, up_a = np.asarray(los), np.asarray(ups)
    singleton = (lo_a == up_a).all(axis=-1)  # (R, T+1)
    tau = np.where(singleton.any(axis=1), np.argmax(singleton, axis=1), t_len)
    rows = np.arange(n_reps)
    covered_at_tau = np.all(
        (lo_a[rows, tau] <= true_ranks[None, :]) & (true_ranks[None, :] <= up_a[rows, tau]),
        axis=1,
    )
    assert float(1.0 - covered_at_tau.mean()) <= ALPHA + VIOL_TOL


def test_rcs_finite_benchmark_coverage_under_random_order() -> None:
    """Sampling model (F): coverage over seeded random evaluation orders.

    A fixed 60-item benchmark with an item-difficulty factor (dependent
    scores across models on every item); the true abilities are the column
    means and the only randomness is the item order (as in the paper's
    finite-benchmark model, the offsets of their Eq. (3) handle the sampling
    without replacement).
    """
    rng = np.random.default_rng(SEED + 11)
    n_items, m = 60, 4
    theta_target = np.array([0.62, 0.55, 0.48, 0.41])
    c_i = rng.standard_normal(n_items) * 0.08
    c_i -= c_i.mean()
    p = np.clip(theta_target[None, :] + c_i[:, None], 0.02, 0.98)
    bench = (rng.random((n_items, m)) < p).astype(float)
    theta_true = bench.mean(axis=0)
    true_ranks = _true_ranks(theta_true)
    n_perm = 60
    los, ups, fwer_hits = [], [], 0
    for _ in range(n_perm):
        perm = rng.permutation(n_items)
        res = rank_confidence_sequence(bench[perm], alpha=ALPHA, method="exact", sampling="finite")
        los.append(res.rank_lower)
        ups.append(res.rank_upper)
        fwer_hits += int(bool(check_dominance_fwer(res.dominance, theta_true)["ever_false"]))
    cov = check_rank_time_uniform_coverage(np.asarray(los), np.asarray(ups), true_ranks)
    assert float(cov["violation_rate"]) <= ALPHA + VIOL_TOL
    assert fwer_hits / n_perm <= ALPHA + VIOL_TOL


def test_rcs_fwer_with_tied_abilities() -> None:
    """Theorem 4.2 Eq. (8): never certify j > l when theta_j == theta_l.

    Two tie groups (0.6, 0.6, 0.45, 0.45): any certified dominance inside a
    tie group is a false dominance; rank-set coverage of the tied true ranks
    (1, 1, 3, 3) must hold at the same time.
    """
    rng = np.random.default_rng(SEED + 12)
    theta = np.array([0.6, 0.6, 0.45, 0.45])
    true_ranks = _true_ranks(theta)
    assert true_ranks.tolist() == [1, 1, 3, 3]
    n_reps, t_len = 80, 80
    panels = _copula_binary_panels(rng, (n_reps, t_len), theta, 0.5)
    hits, los, ups, sets = 0, [], [], []
    for r in range(n_reps):
        res = rank_confidence_sequence(panels[r], alpha=ALPHA, method="exact")
        hits += int(bool(check_dominance_fwer(res.dominance, theta)["ever_false"]))
        los.append(res.rank_lower)
        ups.append(res.rank_upper)
        assert res.rank_sets is not None
        sets.append(res.rank_sets)
    assert hits / n_reps <= ALPHA + VIOL_TOL
    cov = check_rank_time_uniform_coverage(np.asarray(los), np.asarray(ups), true_ranks)
    assert float(cov["violation_rate"]) <= ALPHA + VIOL_TOL
    cov_sets = check_rank_set_coverage(np.asarray(sets), true_ranks)
    assert float(cov_sets["violation_rate"]) <= ALPHA + VIOL_TOL


def test_rcs_early_stopping_stabilization_and_dominance_recovery() -> None:
    """Efficiency demo: rank sets stabilize well before the full sample.

    One dominant model (theta = 0.9 vs 0.5/0.45/0.4): the median
    stabilization time of the FULL interval profile is <= 0.6 * T (observed
    ~45/120 in calibration), the dominant model is certified rank 1 at T in
    >= 90% of replicates, and coverage at the data-dependent stabilization
    time holds (any stopping rule is allowed).
    """
    rng = np.random.default_rng(SEED + 13)
    theta = np.array([0.9, 0.5, 0.45, 0.4])
    true_ranks = _true_ranks(theta)
    n_reps, t_len = 40, 120
    panels = _copula_binary_panels(rng, (n_reps, t_len), theta, 0.3)
    stabs, rank1_hits, los, ups = [], 0, [], []
    for r in range(n_reps):
        res = rank_confidence_sequence(panels[r], alpha=ALPHA, method="exact")
        lo, up = res.rank_lower, res.rank_upper
        los.append(lo)
        ups.append(up)
        same = ((lo == lo[-1][None, :]) & (up == up[-1][None, :])).all(axis=1)
        stabs.append(int(np.argmax(same)) if same.any() else t_len)
        assert res.rank_sets is not None
        if res.rank_sets[-1, 0, 0] and not res.rank_sets[-1, 0, 1:].any():
            rank1_hits += 1
    stabs_arr = np.asarray(stabs)
    assert float(np.median(stabs_arr)) <= 0.6 * t_len
    assert stabs_arr.min() < t_len  # at least one replicate stops strictly early
    assert rank1_hits / n_reps >= 0.9
    lo_a, up_a = np.asarray(los), np.asarray(ups)
    rows = np.arange(n_reps)
    covered = np.all(
        (lo_a[rows, stabs_arr] <= true_ranks[None, :])
        & (true_ranks[None, :] <= up_a[rows, stabs_arr]),
        axis=1,
    )
    assert float(1.0 - covered.mean()) <= ALPHA + VIOL_TOL


def test_bbedge_anytime_fwer_under_dependent_null() -> None:
    """Theorem 1: FWER over ALL replicates and directions under dependence.

    Global null (every theta = 0.5, every direction a true null), a common
    item-difficulty factor inside each replicate (kappa = 0.35) and one
    single block per replicate, the paper's Remark 1 fallback that allows
    ARBITRARY within-replicate dependence. 40 seeded runs x R = 25
    replicates x L = 4 models (12 directions).
    """
    rng = np.random.default_rng(SEED + 14)
    n_models, n_items, n_rep, runs = 4, 40, 25, 40
    theta = np.full(n_models, 0.5)
    delta = theta[:, None] - theta[None, :]
    hits = 0
    for _ in range(runs):
        panels = _copula_binary_panels(rng, (n_rep, n_items), theta, 0.35)
        res = bb_edge_certify(panels, tau=0.0, alpha=ALPHA)
        hits += int(bool(check_edge_fwer(res.edges_closed, delta, tau=0.0)["ever_false"]))
    assert hits / runs <= ALPHA + VIOL_TOL


def test_bbedge_dominance_recovery_rank_coverage_and_topk() -> None:
    """Power side: a dominant model's edges, rank 1, and Top-1 certificate.

    L = 3, theta = (0.8, 0.45, 0.4), item-factor dependence, R = 50
    replicates, 20 seeded runs. Asserts (a) both 0->l edges ever certified in
    >= 80% of runs (observed 20/20 in calibration), (b) simultaneous rank
    interval coverage of the true ranks (1, 2, 3) at every replicate
    (Corollary 1) with violation rate <= alpha + tol, (c) pilot Top-1
    certification in >= 80% of runs with an always-correct candidate.
    """
    rng = np.random.default_rng(SEED + 15)
    n_items, n_rep, runs = 40, 50, 20
    theta = np.array([0.8, 0.45, 0.4])
    delta = theta[:, None] - theta[None, :]
    true_ranks = _true_ranks(theta)
    full_hits, cov_violations, topk_hits, wrong_candidate = 0, 0, 0, 0
    for _ in range(runs):
        panels = _copula_binary_panels(rng, (n_rep, n_items), theta, 0.35)
        res = bb_edge_certify(panels, tau=0.0, alpha=ALPHA)
        ever = res.edges_closed.any(axis=0)
        if ever[0, 1] and ever[0, 2]:
            full_hits += 1
        cov = check_rank_time_uniform_coverage(res.rank_lower, res.rank_upper, true_ranks)
        cov_violations += int(float(cov["violation_rate"]) > 0.0)
        assert not check_edge_fwer(res.edges_closed, delta, tau=0.0)["ever_false"]
        pilot = _copula_binary_panels(rng, (3, n_items), theta, 0.35)
        tk = bb_edge_topk_certify(pilot, panels, 1, tau=0.0, alpha=ALPHA)
        if tk.certified:
            topk_hits += 1
            if tk.candidate.tolist() != [0]:
                wrong_candidate += 1
    assert full_hits / runs >= 0.8
    assert cov_violations / runs <= ALPHA + VIOL_TOL
    assert topk_hits / runs >= 0.8
    assert wrong_candidate == 0


def test_checker_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="matching"):
        check_rank_time_uniform_coverage(np.ones((3, 2)), np.ones((4, 2)), np.array([1, 1]))
    with pytest.raises(ValueError, match="true_ranks"):
        check_rank_time_uniform_coverage(np.ones((3, 2)), np.ones((3, 2)), np.array([1, 5]))
    with pytest.raises(ValueError, match="membership"):
        check_rank_set_coverage(np.zeros((3, 2, 3), dtype=bool), np.array([1, 1]))
    with pytest.raises(ValueError, match="theta"):
        check_dominance_fwer(np.zeros((3, 2, 2), dtype=bool), np.ones(3))
    with pytest.raises(ValueError, match="delta_true"):
        check_edge_fwer(np.zeros((3, 2, 2), dtype=bool), np.ones((3, 3)))


def test_result_dataclass_shapes_and_metadata() -> None:
    rng = np.random.default_rng(SEED + 16)
    res = rank_confidence_sequence(rng.random((15, 3)), alpha=ALPHA, method="exact")
    assert isinstance(res, RankConfidenceSequenceResult)
    assert res.n_models == 3 and res.n_times == 16
    assert res.method == "exact" and res.sampling == "superpopulation"
    panels = rng.random((4, 8, 3))
    bb = bb_edge_certify(panels, alpha=ALPHA)
    assert isinstance(bb, BBEdgeResult)
    assert bb.n_models == 3 and bb.n_replicates == 4
    assert bb.e_values.shape == (4, 6)
    assert bb.directions == tuple(direction_pairs(3))
    assert np.all(bb.thresholds >= 1.0 / ALPHA - 1e-12)
