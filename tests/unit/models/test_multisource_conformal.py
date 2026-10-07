"""MS-RLCP (arXiv:2609.14531) -- SYNTHETIC correctness tests.

Seeded simulations only: kernel forms vs hand-computed values (MS paper
eq. 2.4), the eq. 2.5 weighted score distribution, reduction pins against the
repo's localized-conformal machinery (wide-kernel limit == split conformal ==
``LocalizedCQR``; w_inf = sum(w)/n == ``localized_conformal_quantile``),
single-source MS-RLCP == RLCP, Alg. 1 source selection and its lowest-index
tie rule, the Lemma 3.1 envelope identity max_k ||g_k||_inf == B, the
Theorem 3.3 bound on a hand-computed box-kernel example, gap honesty in the
two-bump bench (degradation flagged via representation term, vacuity and
poor-representation rates, never hidden), determinism, and fail-closed edges.
Correctness material, never market evidence. No Sharpe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.localized_conformal import (
    LocalizedCQR,
    localized_conformal_quantile,
    rbf_weights,
)
from quant_fund.models.multisource_conformal import (
    SourceCalibration,
    alignment_threshold,
    bench_ms_rlcp_two_bumps,
    envelope_constants_1d,
    envelope_coverage_bound,
    localization_kernel,
    ms_rlcp_predict,
    rlcp_predict,
    rlcp_quantile,
    rlcp_scores,
)


def _gauss(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - mu) / sd
    return np.asarray(np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi)), dtype=float)


# ---------------------------------------------------------------------------
# Kernel (MS paper eq. 2.4)
# ---------------------------------------------------------------------------


def test_kernel_gaussian_form_matches_eq_2_4() -> None:
    k = localization_kernel("gaussian", 2.0, 1)
    expected_00 = 1.0 / (math.sqrt(2.0 * math.pi) * 2.0)
    assert k.h_rows(np.array([[1.0]]), np.array([[1.0]]))[0] == pytest.approx(expected_00)
    assert k.h_sup == pytest.approx(expected_00)
    v = k.h_matrix(np.array([[1.0]]), np.array([[0.0]]))[0, 0]
    assert v == pytest.approx(expected_00 * math.exp(-1.0 / 8.0))
    k2 = localization_kernel("gaussian", 1.0, 2)
    assert k2.h_sup == pytest.approx(1.0 / (2.0 * math.pi))
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = k2.h_matrix(a, b)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0 / (2.0 * math.pi))  # zero distance
    assert m[1, 2] == pytest.approx(math.exp(-0.5) / (2.0 * math.pi))  # ||(1,0)-(1,1)|| = 1
    assert m[0, 1] == pytest.approx(math.exp(-0.5) / (2.0 * math.pi))


def test_kernel_box_form_and_closed_ball_boundary() -> None:
    k = localization_kernel("box", 0.5, 1)
    assert k.h_sup == pytest.approx(1.0)  # V_1 = 2, h_sup = 1/(2*0.5)
    x = np.array([[0.0], [0.5], [0.5000001], [1.0]])
    v = k.h_rows(np.zeros((4, 1)), x)
    assert v[0] == 1.0 and v[1] == 1.0  # closed ball: ||dx|| == h included
    assert v[2] == 0.0 and v[3] == 0.0
    k2 = localization_kernel("box", 1.0, 2)
    assert k2.h_sup == pytest.approx(1.0 / math.pi)  # V_2 = pi


def test_kernel_mean_perturb_distance_constants() -> None:
    # Gaussian: h * sqrt(2) * Gamma((d+1)/2) / Gamma(d/2); box: h * d/(d+1).
    kg1 = localization_kernel("gaussian", 0.75, 1)
    assert kg1.mean_perturb_distance == pytest.approx(0.75 * math.sqrt(2.0 / math.pi))
    kg2 = localization_kernel("gaussian", 1.0, 2)
    assert kg2.mean_perturb_distance == pytest.approx(math.sqrt(math.pi / 2.0))
    assert localization_kernel("box", 0.5, 1).mean_perturb_distance == pytest.approx(0.25)
    assert localization_kernel("box", 0.9, 2).mean_perturb_distance == pytest.approx(0.6)


def test_kernel_perturbation_distributions_seeded() -> None:
    kg = localization_kernel("gaussian", 0.7, 1)
    p = kg.perturb(np.zeros((20000, 1)), np.random.default_rng(2))
    assert float(p.std()) == pytest.approx(0.7, abs=0.02)
    assert float(p.mean()) == pytest.approx(0.0, abs=0.02)
    kb = localization_kernel("box", 0.5, 2)
    p2 = kb.perturb(np.zeros((5000, 2)), np.random.default_rng(1))
    r = np.linalg.norm(p2, axis=1)
    assert float(r.max()) <= 0.5 + 1e-12  # uniform on the closed ball
    assert float(r.mean()) == pytest.approx(0.5 * 2.0 / 3.0, abs=0.02)


def test_gaussian_kernel_row_proportional_to_repo_rbf_weights() -> None:
    """Composition pin: H = h_sup * exp(-d^2/(2h^2)) == h_sup * rbf(h' = sqrt(2) h)."""
    rng = np.random.default_rng(9)
    x_cal = rng.normal(size=200)
    h = 0.9
    k = localization_kernel("gaussian", h, 1)
    x_tilde = 0.3
    row = k.h_matrix(np.array([[x_tilde]]), x_cal.reshape(-1, 1))[0]
    assert row == pytest.approx(
        k.h_sup * rbf_weights(x_cal, x_tilde, math.sqrt(2.0) * h), rel=1e-12
    )


def test_kernel_fail_closed() -> None:
    with pytest.raises(ValueError, match="kind"):
        localization_kernel("epanechnikov", 1.0, 1)
    for bad_h in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="bandwidth"):
            localization_kernel("gaussian", bad_h, 1)
    with pytest.raises(ValueError, match="dim"):
        localization_kernel("box", 1.0, 0)
    k = localization_kernel("gaussian", 1.0, 1)
    with pytest.raises(ValueError, match="dimension"):
        k.h_matrix(np.zeros((3, 1)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="dim 1"):
        k.perturb(np.zeros((3, 2)), np.random.default_rng(0))
    with pytest.raises(ValueError, match="same shape"):
        k.h_rows(np.zeros((3, 1)), np.zeros((4, 1)))


def test_alignment_threshold_formula() -> None:
    t = alignment_threshold(1.0, 2, 2000)
    assert t == pytest.approx(2.0 * math.sqrt(2.0 * math.log(2.0 * 2 * 2000) / 2000))
    with pytest.raises(ValueError, match="n_eff"):
        alignment_threshold(1.0, 2, 1)
    with pytest.raises(ValueError, match="n_sources"):
        alignment_threshold(1.0, 0, 10)
    with pytest.raises(ValueError, match="h_sup"):
        alignment_threshold(0.0, 2, 10)


# ---------------------------------------------------------------------------
# Single-source RLCP (MS paper Sec. 2.2 / HB25)
# ---------------------------------------------------------------------------


def test_rlcp_scores_weights_match_eq_2_5() -> None:
    rng = np.random.default_rng(4)
    x_cal = rng.normal(size=60)
    s_cal = np.abs(rng.normal(size=60))
    x_test = rng.normal(size=5)
    k = localization_kernel("gaussian", 0.8, 1)
    x_tilde = x_test + 0.1  # explicit perturbation
    ws = rlcp_scores(x_cal, s_cal, x_test, kernel=k, x_tilde=x_tilde.reshape(-1, 1))
    assert ws.weights.shape == (5, 60)
    assert ws.w_inf.shape == (5,)
    assert np.array_equal(ws.x_tilde, x_tilde.reshape(-1, 1))
    # hand-compute row 0: Z = H(x_test, x~) + sum_i H(x_cal_i, x~)
    h_cal = k.h_matrix(np.array([[x_tilde[0]]]), x_cal.reshape(-1, 1))[0]
    h_test = float(k.h_rows(np.array([[x_test[0]]]), np.array([[x_tilde[0]]]))[0])
    z = h_test + float(np.sum(h_cal))
    assert ws.weights[0] == pytest.approx(h_cal / z, rel=1e-12, abs=0.0)
    assert ws.w_inf[0] == pytest.approx(h_test / z, rel=1e-12)
    total = ws.weights.sum(axis=1) + ws.w_inf
    assert total == pytest.approx(np.ones(5), rel=1e-12)
    assert np.all(ws.weights >= 0.0) and np.all(ws.w_inf > 0.0)
    # locality: the nearest calibration row carries the largest weight
    j = int(np.argmin(np.abs(x_cal - x_tilde[0])))
    assert ws.weights[0, j] == pytest.approx(float(ws.weights[0].max()), abs=1e-15)


def test_rlcp_scores_fail_closed() -> None:
    k = localization_kernel("gaussian", 0.5, 1)
    x_cal = np.array([0.0, 1.0, 2.0])
    s_cal = np.array([0.5, 0.2, 0.9])
    x_test = np.array([1.0])
    with pytest.raises(ValueError, match="per calibration row"):
        rlcp_scores(x_cal, np.array([0.5, 0.2]), x_test, kernel=k, rng=0)
    with pytest.raises(ValueError, match="finite"):
        rlcp_scores(x_cal, np.array([0.5, np.nan, 0.9]), x_test, kernel=k, rng=0)
    with pytest.raises(ValueError, match="dimension"):
        rlcp_scores(x_cal, s_cal, np.zeros((1, 2)), kernel=k, rng=0)
    with pytest.raises(ValueError, match="same shape as x_test"):
        rlcp_scores(x_cal, s_cal, x_test, kernel=k, x_tilde=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="rng"):
        rlcp_scores(x_cal, s_cal, x_test, kernel=k)  # no rng, no x_tilde
    with pytest.raises(ValueError, match="non-empty"):
        rlcp_scores(np.array([]), np.array([]), x_test, kernel=k, rng=0)
    # box kernel: user-supplied x_tilde outside every support -> Z = 0
    kb = localization_kernel("box", 0.5, 1)
    with pytest.raises(ValueError, match="vanish"):
        rlcp_scores(
            np.array([0.0, 0.1]),
            np.array([1.0, 2.0]),
            np.array([0.0]),
            kernel=kb,
            x_tilde=np.array([[50.0]]),
        )


def test_rlcp_quantile_wide_kernel_reduces_to_split_conformal() -> None:
    """h -> inf pins RLCP to the repo's split/localized-conformal machinery."""
    rng = np.random.default_rng(5)
    n, m, alpha = 105, 7, 0.10
    x_cal = rng.normal(size=n) * 1.5
    y_cal = rng.normal(size=n)
    scores = np.abs(y_cal)  # CQR scores with lo = hi = 0
    x_test = rng.normal(size=m) * 1.5
    k_wide = localization_kernel("gaussian", 1e4, 1)
    pred = rlcp_predict(x_cal, scores, x_test, alpha=alpha, kernel=k_wide, x_tilde=x_test)
    q_split = conformal_quantile(scores, alpha)
    q_loc = localized_conformal_quantile(scores, np.ones(n), alpha)
    assert q_split == q_loc
    assert np.all(pred.qhat == q_split)  # exact element equality, not approx
    model = LocalizedCQR(alpha, bandwidth=1e4).calibrate(y_cal, np.zeros(n), np.zeros(n), x_cal)
    lo_l, hi_l = model.predict_sets(np.zeros(m), np.zeros(m), x_test)
    assert np.array_equal((hi_l - lo_l) / 2.0, pred.qhat)


def test_rlcp_quantile_matches_repo_localized_machinery_at_classical_atom() -> None:
    """w_inf = sum(w)/n reproduces ``localized_conformal_quantile`` exactly."""
    rng = np.random.default_rng(8)
    scores = np.abs(rng.normal(size=80))
    for _ in range(5):
        u = rng.random(80) + 0.05
        w_inf = float(np.sum(u)) / 80.0
        assert rlcp_quantile(scores, u, w_inf, 0.10) == localized_conformal_quantile(
            scores, u, 0.10
        )


def test_rlcp_quantile_inf_atom_scale_and_monotonicity() -> None:
    s = np.array([0.5, 1.0, 1.5])
    w = np.array([0.1, 0.1, 0.1])
    assert rlcp_quantile(s, w, 0.7, 0.1) == float("inf")  # +inf atom mass > alpha
    assert rlcp_quantile(s, w, 0.0, 0.5) == 1.0  # w_inf = 0: plain weighted median
    # scale invariance: (w, w_inf) and (c w, c w_inf) give the same qhat
    q1 = rlcp_quantile(s, w, 0.05, 0.2)
    q2 = rlcp_quantile(s, 3.7 * w, 3.7 * 0.05, 0.2)
    assert q1 == q2
    # monotone non-increasing in alpha (levels 1 - alpha shrink)
    qs = [rlcp_quantile(s, np.ones(3), 1.0 / 3.0, a) for a in (0.05, 0.3, 0.6, 0.9)]
    assert all(x >= y for x, y in zip(qs, qs[1:], strict=False))


def test_rlcp_quantile_scalar_matches_rowwise_bitwise() -> None:
    rng = np.random.default_rng(3)
    x_cal = rng.normal(size=90)
    s_cal = np.abs(rng.normal(size=90))
    x_test = rng.normal(size=6)
    ws = rlcp_scores(x_cal, s_cal, x_test, kernel=localization_kernel("gaussian", 0.7, 1), rng=3)
    rows = ws.quantile(0.10)
    for j in range(6):
        scalar = rlcp_quantile(s_cal, ws.weights[j], ws.w_inf[j], 0.10)
        assert rows[j] == scalar  # bit-identical, including +inf rows


def test_rlcp_quantile_fail_closed() -> None:
    s = np.array([0.5, 1.0])
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="non-empty"):
        rlcp_quantile(np.array([]), np.array([]), 0.5, 0.1)
    with pytest.raises(ValueError, match="same length"):
        rlcp_quantile(s, np.ones(3), 0.1, 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        rlcp_quantile(s, np.array([1.0, -0.5]), 0.1, 0.1)
    with pytest.raises(ValueError, match="w_inf"):
        rlcp_quantile(s, w, -0.1, 0.1)
    with pytest.raises(ValueError, match="w_inf"):
        rlcp_quantile(s, w, float("nan"), 0.1)
    with pytest.raises(ValueError, match="total mass"):
        rlcp_quantile(s, np.zeros(2), 0.0, 0.1)
    for bad in (0.0, 1.0, -0.3, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            rlcp_quantile(s, w, 0.1, bad)


def test_rlcp_predict_locality_determinism_and_coverage_flag() -> None:
    """Heteroskedastic cal (repo fixture style): local qhat tracks local scale."""
    rng = np.random.default_rng(21)
    n_cal = 800
    x_cal = rng.choice(np.array([0.4, 2.0]), size=n_cal)
    s_cal = np.abs(rng.normal(0.0, x_cal))
    k = localization_kernel("gaussian", 0.15, 1)
    x_test = np.array([0.4, 2.0])
    p1 = rlcp_predict(x_cal, s_cal, x_test, alpha=0.10, kernel=k, rng=11)
    p2 = rlcp_predict(x_cal, s_cal, x_test, alpha=0.10, kernel=k, rng=11)
    assert np.array_equal(p1.qhat, p2.qhat) and np.array_equal(p1.x_tilde, p2.x_tilde)
    q_lo, q_hi = float(p1.qhat[0]), float(p1.qhat[1])
    assert q_hi > 2.0 * q_lo  # localized to the volatility atom
    assert q_lo == pytest.approx(0.4 * 1.64, rel=0.25)
    assert q_hi == pytest.approx(2.0 * 1.64, rel=0.25)
    assert p1.n_vacuous == 0 and p1.covered is None
    # coverage flag: +inf qhat covers everything; finite qhat compares scores
    p3 = rlcp_predict(
        x_cal,
        s_cal,
        x_test,
        alpha=0.10,
        kernel=k,
        rng=11,
        test_scores=np.array([q_lo - 1e-9, q_hi + 1e-9]),
    )
    assert p3.covered is not None
    assert p3.covered[0] == 1.0 and p3.covered[1] == 0.0
    with pytest.raises(ValueError, match="per test row"):
        rlcp_predict(
            x_cal, s_cal, x_test, alpha=0.10, kernel=k, rng=11, test_scores=np.array([0.1])
        )


# ---------------------------------------------------------------------------
# MS-RLCP (MS paper Alg. 1)
# ---------------------------------------------------------------------------


def _two_bump_sources(
    rng: np.random.Generator,
    n_train: int = 300,
    n_cal: int = 300,
    sep: float = 3.0,
    std: float = 0.7,
) -> tuple[list[SourceCalibration], float, float]:
    def draw(mu: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = mu + std * rng.standard_normal(n)
        sig = 0.4 + 0.22 * np.abs(x)
        return x, sig * rng.standard_normal(n)

    xta, yta = draw(-sep, n_train)
    xtb, ytb = draw(sep, n_train)
    xca, yca = draw(-sep, n_cal)
    xcb, ycb = draw(sep, n_cal)
    ma, mb = float(np.mean(yta)), float(np.mean(ytb))
    return (
        [
            SourceCalibration(xta, xca, np.abs(yca - ma)),
            SourceCalibration(xtb, xcb, np.abs(ycb - mb)),
        ],
        ma,
        mb,
    )


def test_ms_rlcp_single_source_reduces_to_rlcp_exactly() -> None:
    rng = np.random.default_rng(6)
    x_cal = rng.normal(size=120)
    s_cal = np.abs(rng.normal(size=120))
    x_train = rng.normal(size=40)
    x_test = rng.normal(size=9)
    k = localization_kernel("gaussian", 0.8, 1)
    base = rlcp_predict(x_cal, s_cal, x_test, alpha=0.10, kernel=k, rng=13)
    src = SourceCalibration(x_train, x_cal, s_cal)
    ms = ms_rlcp_predict([src], x_test, alpha=0.10, kernel=k, x_tilde=base.x_tilde)
    assert np.all(ms.selected == 0)
    assert np.array_equal(ms.qhat, base.qhat)
    assert np.array_equal(ms.w_inf, base.w_inf)
    assert ms.n_vacuous == base.n_vacuous


def test_ms_rlcp_selects_aligned_source_and_flags_gap() -> None:
    rng = np.random.default_rng(17)
    sources, _, _ = _two_bump_sources(rng)
    k = localization_kernel("gaussian", 0.75, 1)
    x_in = np.concatenate(
        [-3.0 + 0.7 * rng.standard_normal(80), 3.0 + 0.7 * rng.standard_normal(80)]
    )
    x_gap = 0.5 * rng.standard_normal(80)
    p_in = ms_rlcp_predict(sources, x_in, alpha=0.10, kernel=k, rng=31)
    p_gap = ms_rlcp_predict(sources, x_gap, alpha=0.10, kernel=k, rng=32)
    assert p_in.alignment.shape == (160, 2)
    # region A (first 80) -> source 0; region B -> source 1
    assert float(np.mean(p_in.selected[:80] == 0)) >= 0.95
    assert float(np.mean(p_in.selected[80:] == 1)) >= 0.95
    # gap: selection near a coin flip, representation flag fires almost always
    frac_a = float(np.mean(p_gap.selected == 0))
    assert 0.2 <= frac_a <= 0.8
    assert float(np.mean(p_gap.poorly_represented)) >= 0.9
    assert float(np.mean(p_in.poorly_represented)) <= 0.5
    # flag semantics: max alignment <= t, t shared and equal to Thm. 3.3 value
    assert p_gap.alignment_threshold == p_in.alignment_threshold
    assert np.array_equal(
        p_gap.poorly_represented, p_gap.alignment.max(axis=1) <= p_gap.alignment_threshold
    )
    assert p_gap.alignment_threshold == pytest.approx(alignment_threshold(k.h_sup, 2, 300))
    # gap degrades informativeness: more vacuous sets than in-source
    assert p_gap.n_vacuous > p_in.n_vacuous


def test_ms_rlcp_ties_resolve_to_lowest_index() -> None:
    rng = np.random.default_rng(19)
    x_tr = rng.normal(size=100)
    x_cal = rng.normal(size=100)
    s_cal = np.abs(rng.normal(size=100))
    dup = [SourceCalibration(x_tr, x_cal, s_cal), SourceCalibration(x_tr, x_cal, s_cal)]
    p = ms_rlcp_predict(
        dup,
        np.array([[0.0], [1.0], [-2.0]]),
        alpha=0.1,
        kernel=localization_kernel("gaussian", 0.7, 1),
        rng=5,
    )
    assert np.all(p.selected == 0)  # identical alignments -> fixed lowest-index rule
    assert np.all(p.alignment[:, 0] == p.alignment[:, 1])


def test_ms_rlcp_determinism_and_batch_consistency() -> None:
    rng = np.random.default_rng(23)
    sources, _, _ = _two_bump_sources(rng, n_train=120, n_cal=120)
    k = localization_kernel("box", 0.6, 1)
    x_test = np.linspace(-4.0, 4.0, 25)
    a = ms_rlcp_predict(sources, x_test, alpha=0.1, kernel=k, rng=77)
    b = ms_rlcp_predict(sources, x_test, alpha=0.1, kernel=k, rng=77)
    assert np.array_equal(a.qhat, b.qhat) and np.array_equal(a.selected, b.selected)
    assert np.array_equal(a.poorly_represented, b.poorly_represented)
    # box kernel: qhat is a selected calibration score or +inf, never invented
    finite = a.qhat[np.isfinite(a.qhat)]
    all_scores = np.concatenate([s.cal_scores for s in sources])
    assert np.all(np.isin(np.round(finite, 12), np.round(all_scores, 12)))


def test_ms_rlcp_fail_closed() -> None:
    rng = np.random.default_rng(27)
    sources, _, _ = _two_bump_sources(rng, n_train=50, n_cal=50)
    k = localization_kernel("gaussian", 0.7, 1)
    x_test = rng.normal(size=5)
    with pytest.raises(ValueError, match="at least one source"):
        ms_rlcp_predict([], x_test, alpha=0.1, kernel=k, rng=0)
    with pytest.raises(ValueError, match="alpha"):
        ms_rlcp_predict(sources, x_test, alpha=0.0, kernel=k, rng=0)
    with pytest.raises(ValueError, match="rng"):
        ms_rlcp_predict(sources, x_test, alpha=0.1, kernel=k)
    with pytest.raises(ValueError, match="feature dimension"):
        ms_rlcp_predict(sources, rng.normal(size=(5, 2)), alpha=0.1, kernel=k, rng=0)
    with pytest.raises(ValueError, match="per test row"):
        ms_rlcp_predict(sources, x_test, alpha=0.1, kernel=k, rng=0, test_scores=np.zeros(4))
    thin = SourceCalibration(np.array([0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="n_eff"):
        ms_rlcp_predict([thin], x_test, alpha=0.1, kernel=k, rng=0)
    with pytest.raises(ValueError, match="per x_cal row"):
        SourceCalibration(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        SourceCalibration(np.zeros((3, 1)), np.full((3, 1), np.nan), np.zeros(3))


# ---------------------------------------------------------------------------
# Envelope (MS paper Def. 1, Lemma 3.1, Thm. 3.3)
# ---------------------------------------------------------------------------


def test_envelope_constants_two_bumps_and_lemma_3_1() -> None:
    f_a = lambda x: _gauss(x, -3.0, 0.7)  # noqa: E731
    f_b = lambda x: _gauss(x, 3.0, 0.7)  # noqa: E731

    def f_mix(x: np.ndarray) -> np.ndarray:
        return np.asarray(0.5 * (f_a(x) + f_b(x)), dtype=float)

    env = envelope_constants_1d([f_a, f_b], f_mix, lo=-8.6, hi=8.6, n_grid=4001)
    # B = int max(f_a, f_b) = P(X_a <= 0) + P(X_b >= 0) ~= 2 for symmetric bumps
    assert env.b_envelope == pytest.approx(2.0, abs=1e-3)
    # Lemma 3.1: max_k ||g_k||_inf == B
    assert env.g_source_sup == pytest.approx(env.b_envelope, rel=1e-3)
    # mixture test density: sup g_test at the crossing x = 0 equals B
    assert env.g_test_sup == pytest.approx(2.0, abs=1e-2)
    assert env.lipschitz > 0.0 and np.isfinite(env.lipschitz)
    assert env.n_grid == 4001
    # test density equal to one source: sup g == B as well
    env_a = envelope_constants_1d([f_a, f_b], f_a, lo=-8.6, hi=8.6, n_grid=4001)
    assert env_a.g_test_sup == pytest.approx(env_a.b_envelope, rel=1e-3)


def test_envelope_constants_fail_closed() -> None:
    u1 = lambda x: np.where((x >= -2.0) & (x <= -1.0), 1.0, 0.0)  # noqa: E731
    u2 = lambda x: np.where((x >= 1.0) & (x <= 2.0), 1.0, 0.0)  # noqa: E731
    u_gap = lambda x: np.where((x >= -1.0) & (x <= 1.0), 0.5, 0.0)  # noqa: E731
    with pytest.raises(ValueError, match="Assumption 1"):
        envelope_constants_1d([u1, u2], u_gap, lo=-3.0, hi=3.0, n_grid=601)
    ok = lambda x: _gauss(x, 0.0, 1.0)  # noqa: E731
    with pytest.raises(ValueError, match="non-empty"):
        envelope_constants_1d([], ok, lo=-3.0, hi=3.0)
    with pytest.raises(ValueError, match="lo < hi"):
        envelope_constants_1d([ok], ok, lo=3.0, hi=-3.0)
    with pytest.raises(ValueError, match="n_grid"):
        envelope_constants_1d([ok], ok, lo=-3.0, hi=3.0, n_grid=2)
    with pytest.raises(ValueError, match="per grid point"):
        envelope_constants_1d([lambda x: np.ones(3)], ok, lo=-3.0, hi=3.0, n_grid=101)
    with pytest.raises(ValueError, match="non-negative"):
        envelope_constants_1d(
            [lambda x: -np.ones_like(np.asarray(x, float))], ok, lo=-3.0, hi=3.0, n_grid=101
        )
    with pytest.raises(ValueError, match="finite"):
        envelope_constants_1d(
            [lambda x: np.full_like(np.asarray(x, float), np.nan)], ok, lo=-3.0, hi=3.0, n_grid=101
        )
    with pytest.raises(ValueError, match="vanish"):
        envelope_constants_1d(
            [lambda x: np.zeros_like(np.asarray(x, float))], ok, lo=-3.0, hi=3.0, n_grid=101
        )


def test_envelope_bound_hand_example_box_kernel() -> None:
    """Deterministic hand-computed Thm. 3.3 terms: uniform sources, box kernel.

    h = 0.5 (||H||_inf = 1, E||X - X~|| = h/2), K = 2, n_eff = 2000 ->
    t = 2 sqrt(2 ln(8000)/2000) = 0.189602. Interior perturbed points see
    mu_k ~= 1/2 > t (never flagged); points at distance > h from both sources
    see mu_k = 0 (always flagged). With L = 0 the bound is exactly
    0.9 - 1/2000 - rep.
    """
    kb = localization_kernel("box", 0.5, 1)
    s1 = np.linspace(-2.0, -1.0, 2000)
    s2 = np.linspace(1.0, 2.0, 2000)
    common = dict(alpha=0.1, kernel=kb, lipschitz=0.0, g_test_sup=1.0, b_envelope=2.0)
    b_in = envelope_coverage_bound(
        **common, x_test=np.array([[-1.5], [1.5]]), source_train=[s1, s2], rng=4
    )
    b_far = envelope_coverage_bound(
        **common, x_test=np.array([[7.0], [-7.0]]), source_train=[s1, s2], rng=4
    )
    assert b_in.alignment_threshold == pytest.approx(
        2.0 * math.sqrt(2.0 * math.log(2.0 * 2 * 2000) / 2000)
    )
    assert b_in.representation_term == 0.0
    assert b_far.representation_term == 1.0
    assert b_in.finite_sample_term == pytest.approx(1.0 / 2000.0)
    assert b_in.localization_term == 0.0  # L = 0
    assert b_in.mean_perturb_distance == pytest.approx(0.25)
    assert b_in.nominal == pytest.approx(0.9)
    assert b_in.bound == pytest.approx(0.9 - 1.0 / 2000.0)
    assert b_in.vacuous is False
    assert b_far.bound == pytest.approx(0.9 - 1.0 / 2000.0 - 1.0)
    assert b_far.vacuous is True  # degradation flagged, not hidden
    assert b_in.n_eff == 2000 and b_in.n_sources == 2
    # seed-independence: the flag events are deterministic here
    b_in2 = envelope_coverage_bound(
        **common, x_test=np.array([[-1.5], [1.5]]), source_train=[s1, s2], rng=99
    )
    assert b_in2.bound == b_in.bound


def test_envelope_bound_monotone_in_oracle_constants() -> None:
    kb = localization_kernel("box", 0.5, 1)
    s1 = np.linspace(-2.0, -1.0, 500)
    x_test = np.array([[-1.5], [1.5], [0.0]])
    base = envelope_coverage_bound(
        alpha=0.1,
        kernel=kb,
        lipschitz=0.5,
        g_test_sup=1.0,
        b_envelope=2.0,
        x_test=x_test,
        source_train=[s1],
        rng=2,
    )
    steeper = envelope_coverage_bound(
        alpha=0.1,
        kernel=kb,
        lipschitz=5.0,
        g_test_sup=1.0,
        b_envelope=2.0,
        x_test=x_test,
        source_train=[s1],
        rng=2,
    )
    assert steeper.localization_term == pytest.approx(10.0 * base.localization_term)
    assert steeper.bound < base.bound
    shifted = envelope_coverage_bound(
        alpha=0.1,
        kernel=kb,
        lipschitz=0.5,
        g_test_sup=50.0,
        b_envelope=2.0,
        x_test=x_test,
        source_train=[s1],
        rng=2,
    )
    assert shifted.localization_term > base.localization_term
    assert shifted.bound < base.bound
    assert base.representation_term == shifted.representation_term  # same features/seed
    # localization term scales with the kernel's mean perturbation distance
    wide = localization_kernel("box", 1.0, 1)
    b_wide = envelope_coverage_bound(
        alpha=0.1,
        kernel=wide,
        lipschitz=0.5,
        g_test_sup=1.0,
        b_envelope=2.0,
        x_test=x_test,
        source_train=[s1],
        rng=2,
    )
    assert b_wide.mean_perturb_distance == pytest.approx(2.0 * base.mean_perturb_distance)
    assert b_wide.localization_term == pytest.approx(2.0 * base.localization_term)


def test_envelope_bound_fail_closed() -> None:
    kb = localization_kernel("box", 0.5, 1)
    s1 = np.linspace(-2.0, -1.0, 10)
    x_test = np.array([[-1.5]])
    good = dict(
        alpha=0.1,
        kernel=kb,
        lipschitz=0.5,
        g_test_sup=1.0,
        b_envelope=2.0,
        x_test=x_test,
        source_train=[s1],
        rng=0,
    )
    with pytest.raises(ValueError, match="n_eff"):
        envelope_coverage_bound(**{**good, "source_train": [np.linspace(-2.0, -1.0, 1)]})
    with pytest.raises(ValueError, match="source_train"):
        envelope_coverage_bound(**{**good, "source_train": []})
    with pytest.raises(ValueError, match="lipschitz"):
        envelope_coverage_bound(**{**good, "lipschitz": -1.0})
    with pytest.raises(ValueError, match="g_test_sup"):
        envelope_coverage_bound(**{**good, "g_test_sup": float("nan")})
    with pytest.raises(ValueError, match="b_envelope"):
        envelope_coverage_bound(**{**good, "b_envelope": 0.0})
    with pytest.raises(ValueError, match="alpha"):
        envelope_coverage_bound(**{**good, "alpha": 1.5})
    with pytest.raises(ValueError, match="feature dimension"):
        envelope_coverage_bound(**{**good, "x_test": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="rng"):
        envelope_coverage_bound(**{k: v for k, v in good.items() if k != "rng"})


# ---------------------------------------------------------------------------
# SYNTHETIC bench: two disjoint-ish bumps, in-source vs gap honesty
# ---------------------------------------------------------------------------

BENCH_KEYS = {
    "synthetic_dgp",
    "synthetic_claim",
    "synthetic_kernel",
    "synthetic_seed",
    "synthetic_alpha",
    "synthetic_bandwidth",
    "synthetic_bump_std",
    "synthetic_bump_sep",
    "synthetic_gap_std",
    "synthetic_n_sources",
    "synthetic_n_train",
    "synthetic_n_cal",
    "synthetic_n_eff",
    "synthetic_alignment_threshold",
    "synthetic_n_test_in",
    "synthetic_n_test_gap",
    "synthetic_coverage_in_source",
    "synthetic_coverage_in_source_region_a",
    "synthetic_coverage_in_source_region_b",
    "synthetic_mean_width_in_source",
    "synthetic_qhat_median_in_source",
    "synthetic_frac_vacuous_in_source",
    "synthetic_poorly_represented_rate_in_source",
    "synthetic_coverage_gap",
    "synthetic_mean_width_gap_finite",
    "synthetic_qhat_median_gap_finite",
    "synthetic_frac_vacuous_gap",
    "synthetic_poorly_represented_rate_gap",
    "synthetic_selection_rate_a_region_a",
    "synthetic_selection_rate_b_region_b",
    "synthetic_selection_rate_a_gap",
    "synthetic_envelope_b",
    "synthetic_envelope_g_source_sup",
    "synthetic_envelope_g_test_sup_in_source",
    "synthetic_envelope_g_test_sup_gap",
    "synthetic_envelope_lipschitz_in_source",
    "synthetic_envelope_lipschitz_gap",
    "synthetic_mean_perturb_distance",
    "synthetic_localization_term_in_source",
    "synthetic_localization_term_gap",
    "synthetic_representation_term_in_source",
    "synthetic_representation_term_gap",
    "synthetic_bound_in_source",
    "synthetic_bound_gap",
    "synthetic_bound_vacuous_in_source",
    "synthetic_bound_vacuous_gap",
}


def test_bench_two_bumps_coverage_and_gap_honesty() -> None:
    row = bench_ms_rlcp_two_bumps()
    assert set(row) == BENCH_KEYS
    assert row["synthetic_claim"] == "research_metric_only"
    assert row["synthetic_dgp"] == "synthetic_two_bumps"
    assert all("sharpe" not in key.lower() for key in row)
    nominal = 1.0 - float(row["synthetic_alpha"])
    for key, v in row.items():
        if key not in ("synthetic_dgp", "synthetic_claim", "synthetic_kernel"):
            assert np.isfinite(float(v)), key
    # in-source coverage: at nominal within MC tolerance, in both regions
    assert float(row["synthetic_coverage_in_source"]) >= nominal - 0.035
    assert float(row["synthetic_coverage_in_source_region_a"]) >= nominal - 0.05
    assert float(row["synthetic_coverage_in_source_region_b"]) >= nominal - 0.05
    assert float(row["synthetic_coverage_in_source"]) <= 0.98  # not vacuously over-covering
    # selection sanity: aligned source wins in-region; gap is near a coin flip
    assert float(row["synthetic_selection_rate_a_region_a"]) >= 0.9
    assert float(row["synthetic_selection_rate_b_region_b"]) >= 0.9
    assert 0.25 <= float(row["synthetic_selection_rate_a_gap"]) <= 0.75
    # gap honesty: degradation flagged, not hidden
    assert float(row["synthetic_poorly_represented_rate_gap"]) >= (
        float(row["synthetic_poorly_represented_rate_in_source"]) + 0.4
    )
    assert float(row["synthetic_representation_term_gap"]) >= 0.9
    assert float(row["synthetic_representation_term_in_source"]) <= 0.5
    assert (
        float(row["synthetic_frac_vacuous_gap"])
        >= float(row["synthetic_frac_vacuous_in_source"]) + 0.1
    )
    assert float(row["synthetic_bound_gap"]) < float(row["synthetic_bound_in_source"])
    assert float(row["synthetic_bound_vacuous_gap"]) == 1.0  # vacuity surfaced explicitly
    assert float(row["synthetic_localization_term_gap"]) > 100.0 * float(
        row["synthetic_localization_term_in_source"]
    )
    assert float(row["synthetic_envelope_g_test_sup_gap"]) > 100.0 * float(
        row["synthetic_envelope_g_test_sup_in_source"]
    )
    # Lemma 3.1 through the bench's oracle grid constants
    assert float(row["synthetic_envelope_g_source_sup"]) == pytest.approx(
        float(row["synthetic_envelope_b"]), rel=1e-3
    )
    # widths are informative in-source (localization keeps sets tight)
    assert float(row["synthetic_mean_width_in_source"]) > 0.0
    assert float(row["synthetic_qhat_median_in_source"]) < 3.0


def test_bench_deterministic_and_seed_sensitive() -> None:
    a = bench_ms_rlcp_two_bumps(seed=13)
    b = bench_ms_rlcp_two_bumps(seed=13)
    assert a == b
    c = bench_ms_rlcp_two_bumps(seed=14)
    assert c["synthetic_seed"] == 14.0
    # different data, same qualitative story: in-source coverage still valid
    assert float(c["synthetic_coverage_in_source"]) >= 1.0 - float(c["synthetic_alpha"]) - 0.05
    assert float(c["synthetic_representation_term_gap"]) > float(
        c["synthetic_representation_term_in_source"]
    )


def test_bench_second_seed_coverage_holds() -> None:
    row = bench_ms_rlcp_two_bumps(seed=7, n_test_in=800, n_test_gap=400)
    nominal = 1.0 - float(row["synthetic_alpha"])
    assert float(row["synthetic_coverage_in_source"]) >= nominal - 0.035
    assert float(row["synthetic_coverage_gap"]) >= nominal - 0.035
    assert float(row["synthetic_bound_gap"]) < float(row["synthetic_bound_in_source"])


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bench_ms_rlcp_two_bumps(alpha=0.0)
    with pytest.raises(ValueError, match=">= 2"):
        bench_ms_rlcp_two_bumps(n_test_in=1)
    with pytest.raises(ValueError, match="bump_std"):
        bench_ms_rlcp_two_bumps(bump_std=0.0)
    with pytest.raises(ValueError, match="bump_sep"):
        bench_ms_rlcp_two_bumps(bump_sep=-1.0)
    with pytest.raises(ValueError, match="gap_std"):
        bench_ms_rlcp_two_bumps(gap_std=float("nan"))
    with pytest.raises(ValueError, match="bandwidth"):
        bench_ms_rlcp_two_bumps(bandwidth=0.0)
