"""Regression tests for defects found in the models audit (docs/AUDIT_MODELS.md)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.ar_estimation import levinson_durbin
from quant_fund.models.diffusion_index import diffusion_index_forecast, sw_factors
from quant_fund.models.duration import _nll_weibull
from quant_fund.models.factor_models import bai_ng_factors, double_sorted_factors
from quant_fund.models.fgls_ar1 import prais_winsten
from quant_fund.models.gas import gas_vol_fit, gas_vol_forecast
from quant_fund.models.nonlinear_filters import particle_filter
from quant_fund.models.nowcasting import beta_weights, fit_midas
from quant_fund.models.state_space import ou_mle
from quant_fund.models.tail import DrawdownClassifier, ScaledHistoricalTail
from quant_fund.models.var_coint import engle_granger, johansen_test, vecm_fit


def test_engle_granger_rejects_nonfinite_x():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=200))
    x[50] = np.nan
    y = x + rng.normal(size=200)
    with pytest.raises(ValueError):
        engle_granger(y, x)


def test_engle_granger_still_rejects_nonfinite_y():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=200))
    y = x + rng.normal(size=200)
    y[50] = np.inf
    with pytest.raises(ValueError):
        engle_granger(y, x)


def test_johansen_cv_indexed_by_n_minus_r():
    # n=2: under H0 rank<=0 there are n-r=2 independent combinations ->
    # the 5% trace CV must be 9.24, not the n-r=1 entry 3.76.
    rng = np.random.default_rng(1)
    eps = rng.normal(size=(300, 2))
    beta = np.array([1.0, -1.0])
    alpha = np.array([[-0.1], [0.0]])
    y = np.zeros((300, 2))
    for t in range(1, 300):
        y[t] = y[t - 1] + (alpha @ np.array([[beta @ y[t - 1] - 0.0]])).ravel() + eps[t]
    out = johansen_test(y, p=2)
    assert out["cv_trace5"][0] == pytest.approx(9.24)
    assert out["cv_max5"][0] == pytest.approx(11.22)
    # rank = first r whose statistic does not reject; all-rejected -> n.
    assert 0 <= out["rank_trace"][0] <= 2
    # A genuinely cointegrated pair should detect rank 1.
    assert out["rank_trace"][0] == 1.0 or out["rank_max"][0] == 1.0


def test_vecm_gamma_lag_ordering():
    # dy_t = Gamma1 dy_{t-1} + 0 * dy_{t-2} + small EC pull; lag-1 block must
    # land in gamma[0], not be scrambled across lags.
    rng = np.random.default_rng(7)
    n, tt = 2, 600
    g1 = np.diag([0.5, 0.3])
    alpha = np.array([[-0.05], [0.0]])
    beta = np.array([[1.0], [-1.0]])
    dy = np.zeros((tt, n))
    y = np.zeros((tt, n))
    for t in range(2, tt):
        ect = float((beta.T @ y[t - 1]).item())
        dy[t] = g1 @ dy[t - 1] + (alpha * ect).ravel() + 0.1 * rng.normal(size=n)
        y[t] = y[t - 1] + dy[t]
    out = vecm_fit(y, p=3, rank=1)
    gamma = np.asarray(out["gamma"])
    assert gamma.shape == (2, n, n)
    # True Gamma1[1,1] ~ 0.3; the buggy reshape scrambled lag-1/lag-2 rows so
    # gamma[0][1,:] picked up the ~0 lag-2 coefficients instead.
    assert gamma[0][1, 1] == pytest.approx(0.3, abs=0.15)
    assert np.abs(gamma[1]).max() < 0.2


def test_diffusion_index_forecast_origin():
    # A one-step forecast must use the true origin row t = T-1 ([1, F_{T-1},
    # y_T-1...]), not the last fitted row t = T-2.
    rng = np.random.default_rng(3)
    panel = rng.normal(size=(60, 8))
    f = np.asarray(sw_factors(panel, 1)["factors"])[:, 0]
    y = 1.5 + 2.0 * f
    out = diffusion_index_forecast(panel, y, k=1, ar_lags=1, horizon=1)
    fc = float(out["forecast"])
    beta = np.asarray(out["coefs"])
    correct_row = np.array([1.0, f[-1], y[-1]])
    buggy_row = np.array([1.0, f[-2], y[-2]])
    assert fc == pytest.approx(float(correct_row @ beta), abs=1e-12)
    assert abs(fc - float(buggy_row @ beta)) > 1e-8


def test_bai_ng_zero_variance_panel_raises():
    panel = np.full((50, 4), 3.0)
    with pytest.raises(ValueError, match="zero total variance"):
        bai_ng_factors(panel)


def test_levinson_durbin_non_pd_raises():
    # r1 > r0 is impossible for a valid autocovariance; must raise, not
    # silently floor the innovation variance at 1e-12.
    with pytest.raises(ValueError, match="positive-definite"):
        levinson_durbin(np.array([1.0, 2.0, 0.5]), 2)


def test_prais_winsten_rho_outside_unit_interval_raises():
    # Exponentially growing residuals give num/den > 1; the PW transform is
    # invalid there and must fail closed.
    t = np.linspace(0.0, 3.0, 50)
    y = np.exp(t)
    x = np.column_stack([np.random.default_rng(0).normal(size=50)])
    with pytest.raises(ValueError, match="outside"):
        prais_winsten(y, x)


def test_ou_mle_non_stationary_raises():
    # Explosive transition (phi > 1) must raise rather than clamp to 0.999999.
    x = np.exp(np.linspace(0.0, 2.0, 100))
    with pytest.raises(ValueError, match="mean-reverting"):
        ou_mle(x)


def test_gas_fit_exposes_scaling_and_forecast_uses_it():
    rng = np.random.default_rng(11)
    eps = rng.normal(size=400)
    h = np.empty(400)
    h[0] = 1.0
    for t in range(1, 400):
        h[t] = 0.05 + 0.1 * eps[t - 1] ** 2 + 0.85 * h[t - 1]
    y = np.sqrt(h) * eps
    fit = gas_vol_fit(y, dist="gauss", scaling="unit")
    assert fit["unit_scaling"] == 1.0
    f_last = float(np.asarray(fit["f"])[-1])
    y_last = float(y[-1])
    s = (y_last * y_last - f_last) / (2.0 * f_last * f_last)
    u_unit = s * 2.0 * f_last
    expected = float(fit["omega"]) + float(fit["A"]) * u_unit + float(fit["B"]) * f_last
    assert gas_vol_forecast(fit, y_last, dist="gauss") == pytest.approx(expected)


def test_acd_weibull_nll_includes_psi_jacobian():
    # Correct WACD loglik: ln g - g ln(psi*lam) + (g-1) ln(x/(psi*lam))... i.e.
    # -psi coefficient must be -gamma, not -1. For gamma=1 both agree; for
    # gamma=2 the missing (g-1)*ln psi term shifts the value.
    x = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 1.0, 0.95, 1.05, 1.1])
    theta = np.array([0.3, 0.2, 0.5, 2.0])  # omega, alpha, beta, gamma=2
    omega, alpha, beta, gamma = theta
    psi = np.empty(x.size)
    psi[0] = x.mean()  # matches _psi_path's initialization
    for t in range(1, x.size):
        psi[t] = omega + alpha * x[t - 1] + beta * psi[t - 1]
    lam = math.exp(math.lgamma(1.0 + 1.0 / gamma))
    z = x / psi / lam
    # Derived: eps = x/psi, f_x = f_eps(x/psi)/psi =>
    # ln f = ln g - g ln lam - ln psi + (g-1)(ln x - ln psi) - z^g
    #      = ln g - g ln(psi*lam) + (g-1) ln x - z^g.
    expected = np.sum(
        np.log(gamma) - gamma * np.log(psi * lam) + (gamma - 1.0) * np.log(x) - z**gamma
    )
    # The pre-fix code had (g-1) ln x and only -ln psi, i.e. psi weight -1
    # instead of -gamma.
    assert _nll_weibull(theta, x) == pytest.approx(float(-expected))


def test_particle_filter_carries_weights_without_resample():
    # With resample_frac=0 the filter never resamples; weights must be the
    # running product w_t ~ w_{t-1} * p(y_t|x_t), not reset each step.
    n_p = 50
    seed = 4
    q_std = 0.3
    y = np.array([0.8, -0.6, 0.2, 0.5, -0.1])

    out = particle_filter(
        y,
        f=lambda x, rng: x,
        obs_loglik=lambda x, yy: -0.5 * (x - yy) ** 2,
        q_std=q_std,
        n_particles=n_p,
        x0=0.0,
        p0_std=1.0,
        seed=seed,
        resample_frac=0.0,
    )
    rng = np.random.default_rng(seed)
    parts = rng.normal(0.0, 1.0, n_p)
    w = np.full(n_p, 1.0 / n_p)
    means, loglik = [], 0.0
    for t in range(y.size):
        parts = parts + rng.normal(0.0, q_std, n_p)
        ll = -0.5 * (parts - y[t]) ** 2
        mx = ll.max()
        w = w * np.exp(ll - mx)
        tot = w.sum()
        w /= tot
        loglik += mx + math.log(tot)
        means.append(float(w @ parts))
    np.testing.assert_allclose(out["mean"], means, rtol=1e-12, atol=1e-12)
    assert out["loglik"] == pytest.approx(loglik, rel=1e-12, abs=1e-12)


def test_midas_ar_lag_false_actually_optimizes():
    # With ar_lag=False the NaN in yl[0] used to poison every SSE evaluation
    # (0 * NaN), so all starts returned the 1e12 penalty and the optimizer
    # never moved off its seed point.
    rng = np.random.default_rng(0)
    n, k = 200, 6
    x = rng.normal(size=(n, k))
    w_true = beta_weights(k, 1.0, 8.0)
    y = 0.5 + 2.0 * (x @ w_true) + rng.normal(scale=0.1, size=n)
    fit = fit_midas(y, x, k_lags=k, ar_lag=False)
    assert fit["sse"] < 1e12
    assert abs(fit["slope"] - 2.0) < 0.4


def test_scaled_historical_tail_unfitted_raises():
    m = ScaledHistoricalTail()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_var_es(np.ones(3))


def test_drawdown_classifier_unfitted_raises():
    m = DrawdownClassifier()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_proba(np.zeros((3, 2)))


# --- Optimizer-penalty guards -------------------------------------------------
# Each objective returns a 1e12 penalty on invalid regions; the guard is that
# this penalty must never be accepted as a converged optimum.


class _PenaltyResult:
    def __init__(self, x):
        self.x = x
        self.fun = 1e12
        self.success = True


def test_skew_t_fit_rejects_penalty(monkeypatch):
    import quant_fund.models.skew_t as mod

    monkeypatch.setattr(
        mod, "minimize", lambda *a, **k: _PenaltyResult(np.array([8.0, 0.0, 0.0, 0.0]))
    )
    with pytest.raises(ValueError, match="skew-t MLE"):
        mod.skew_t_fit(np.random.default_rng(0).normal(size=100))


def test_joe_fit_rejects_penalty(monkeypatch):
    import quant_fund.models.archimedean_extra as mod

    monkeypatch.setattr(mod, "minimize_scalar", lambda *a, **k: _PenaltyResult(np.array(2.0)))
    u = np.random.default_rng(0).uniform(0.01, 0.99, size=(60, 2))
    with pytest.raises(ValueError, match="Joe copula"):
        mod.joe_fit(u)


def test_whittle_estimators_reject_penalty(monkeypatch):
    import quant_fund.models.long_memory as mod

    monkeypatch.setattr(mod.opt, "minimize_scalar", lambda *a, **k: _PenaltyResult(np.array(0.3)))
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(size=300))
    with pytest.raises(ValueError, match="Whittle"):
        mod.local_whittle(series)
    with pytest.raises(ValueError, match="Whittle"):
        mod.whittle_arfima(series)


def test_dcc_stage2_rejects_penalty(monkeypatch):
    import quant_fund.models.dcc as mod

    monkeypatch.setattr(
        mod.optimize,
        "minimize",
        lambda f, x0, **k: _PenaltyResult(np.asarray(x0, dtype=float)),
    )
    r = np.random.default_rng(0).normal(size=(150, 3))
    with pytest.raises(ValueError, match="DCC stage-2"):
        mod.dcc_fit(r)


def test_svensson_rejects_penalty(monkeypatch):
    import quant_fund.models.term_structure as mod

    monkeypatch.setattr(
        mod.optimize,
        "minimize",
        lambda f, x0, **k: _PenaltyResult(np.asarray(x0, dtype=float)),
    )
    m = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
    y = np.array([2.0, 2.2, 2.5, 2.8, 3.0])
    with pytest.raises(ValueError, match="Svensson"):
        mod.svensson_fit(m, y)


def test_garch_midas_rejects_penalty(monkeypatch):
    import quant_fund.models.garch_midas as mod

    monkeypatch.setattr(
        mod.optimize,
        "minimize",
        lambda f, x0, **k: _PenaltyResult(np.asarray(x0, dtype=float)),
    )
    rng = np.random.default_rng(0)
    r = rng.normal(size=300)
    with pytest.raises(ValueError, match="GARCH-MIDAS"):
        mod.garch_midas_fit(r, block=21, k_lag=5)


def test_figarch_aparch_reject_penalty(monkeypatch):
    import quant_fund.models.garch_ext as mod

    monkeypatch.setattr(
        mod.opt, "minimize", lambda *a, **k: _PenaltyResult(np.asarray(a[1], dtype=float))
    )
    r = np.random.default_rng(0).normal(size=300)
    with pytest.raises(ValueError, match="FIGARCH"):
        mod.fit_figarch(r)
    with pytest.raises(ValueError, match="APARCH"):
        mod.fit_aparch(r)


def test_double_sorted_factors_permutation_invariant_under_ties():
    # tie group straddling a cell boundary must land in one cell regardless
    # of storage order: permuting asset order must not change the output.
    rng = np.random.default_rng(7)
    t, n = 6, 12
    a = rng.normal(size=(t, n))
    b = rng.normal(size=(t, n))
    # inject tie groups that straddle the cuts=3 bin boundaries
    a[:, 3:6] = 0.5
    b[:, 8:11] = -0.2
    r = rng.normal(size=(t, n))
    base = double_sorted_factors(r, a, b, cuts=3)
    for _ in range(20):
        perm = rng.permutation(n)
        out = double_sorted_factors(r[:, perm], a[:, perm], b[:, perm], cuts=3)
        assert np.allclose(base["factor_a"], out["factor_a"])
        assert np.allclose(base["factor_b"], out["factor_b"])
        assert np.allclose(
            np.nan_to_num(base["grid"], nan=-1.0),
            np.nan_to_num(out["grid"], nan=-1.0),
        )


def test_double_sorted_factors_no_asset_dropped():
    # every asset lands in exactly one grid cell — with 1-based midranks the
    # naive rank*cuts//n formula pushes rank n into cell `cuts` and drops it.
    # oracle: scipy rankdata('average') partition vs the returned grid means.
    from scipy.stats import rankdata

    rng = np.random.default_rng(3)
    t, n, cuts = 5, 12, 3
    r = rng.normal(size=(t, n))
    a = rng.normal(size=(t, n))
    b = rng.normal(size=(t, n))
    a[0, :3] = 0.9  # tie group across the top boundary
    out = double_sorted_factors(r, a, b, cuts=cuts)
    for s in range(t):
        qa = ((rankdata(a[s], method="average") - 1.0) * cuts // n).astype(int)
        qb = ((rankdata(b[s], method="average") - 1.0) * cuts // n).astype(int)
        for i in range(cuts):
            for j in range(cuts):
                sel = (qa == i) & (qb == j)
                cell = out["grid"][s, i, j]
                if np.any(sel):
                    assert np.isfinite(cell)
                    assert cell == pytest.approx(float(r[s, sel].mean()))
                else:
                    assert not np.isfinite(cell)


def test_gp_regression_penalty_falls_back_to_defaults(monkeypatch):
    import scipy.optimize as so

    import quant_fund.models.kernel as mod

    monkeypatch.setattr(so, "minimize", lambda *a, **k: _PenaltyResult(np.zeros(3)))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 2))
    y = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=30)
    fit = mod.fit_gp_regression(x, y)
    # fallback to the median-length heuristic, not the penalized point
    assert fit["length"] == pytest.approx(mod._median_length(x))


# --- models-core audit probes -------------------------------------------------
# Adversarial/seeded probes for the second wave of audited modules
# (delayed_aci, extra_tilt_conformal, favar, graded_irt, neural_tpp,
# odd_residual_flows, pair_vine_copula, path_signatures, regime_conformal_var,
# rrc_filter, svi_surface, tda_persistence, viterbi_decode, rolling_conformal,
# rbergomi, fbm, deep_regime_mixture, ivs_diffusion).


def test_viterbi_rejects_nonbinary_encoder_and_degenerate_streams():
    from quant_fund.models.viterbi_decode import conv_encode, viterbi_hard, viterbi_soft

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 40).astype(float)
    # clean encode/decode still works (control)
    np.testing.assert_array_equal(viterbi_hard(conv_encode(bits)), bits)
    # encoder silently truncated 0.7 -> 0 before the fix
    with pytest.raises(ValueError, match="0/1"):
        conv_encode(np.array([0.0, 0.7, 1.0]))
    with pytest.raises(ValueError, match="0/1"):
        conv_encode(np.array([0.0, np.nan, 1.0]))
    # decoders: odd-length streams were silently truncated, non-finite
    # symbols fabricated output bits through a garbage traceback
    enc = conv_encode(bits)
    with pytest.raises(ValueError, match="even"):
        viterbi_hard(enc[:-1])
    with pytest.raises(ValueError, match="0/1"):
        viterbi_hard(np.array([0.0, 0.4, 1.0, 1.0]))
    with pytest.raises(ValueError, match="even"):
        viterbi_soft(enc[:-1])
    bad = enc.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        viterbi_soft(bad)


def test_graded_irt_rejects_fractional_category_codes():
    from quant_fund.models.graded_irt import grm_jml

    rng = np.random.default_rng(1)
    resp = rng.integers(0, 3, size=(8, 4)).astype(float)
    resp[2, 1] = 0.7  # int64 cast used to silently truncate to 0
    with pytest.raises(ValueError, match="integer-valued"):
        grm_jml(resp, n_iter=2)
    resp[4, 0] = np.nan
    with pytest.raises(ValueError, match="integer-valued"):
        grm_jml(resp, n_iter=2)


def test_neural_tpp_rejects_fractional_marks_and_penalty(monkeypatch):
    import quant_fund.models.neural_tpp as mod

    t = np.array([0.1, 0.4, 0.9, 1.5, 2.0])
    marks = np.array([0.0, 0.6, 0.0, 1.0, 1.0])  # 0.6 truncated to 0 pre-fix
    with pytest.raises(ValueError, match="integer-valued"):
        mod.hawkes_intensity_path(t, marks, 0, mu=0.2, alpha=0.5, beta=1.5)

    monkeypatch.setattr(
        mod, "minimize", lambda *a, **k: _PenaltyResult(np.log(np.array([0.2, 0.4, 1.0])))
    )
    with pytest.raises(ValueError, match="hawkes MLE"):
        mod.hawkes_mle_univariate(np.cumsum(np.random.default_rng(0).uniform(size=20)))


def test_tda_vr_pairs_cap_fires_before_materializing(monkeypatch):
    # n=30 dense complex has ~4.5k triangles vs cap 100; the old code built
    # the whole C(n,3) list before checking. A tight cap must raise fast.
    import quant_fund.models.tda_persistence as mod

    monkeypatch.setattr(mod, "_MAX_SIMPLICES", 100)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(30, 2))
    with pytest.raises(ValueError, match="simplices"):
        mod.h1_persistence(pts)
    # filtered complex that fits under the cap still reduces normally
    dgm = mod.h1_persistence(pts, r_max=0.5)
    assert dgm.shape[1] == 2


def test_tda_max_matching_iterative_deep_path():
    # a 1D chain where every augmenting path is ~2n hops: the recursive
    # augment() hit RecursionError on large matchings instead of answering.
    import quant_fund.models.tda_persistence as mod

    n = 400
    adj = [[i] + ([i - 1] if i > 0 else []) for i in range(n)]
    match = mod._max_matching(adj, n)
    # perfect matching exists: each right vertex lands some left vertex
    assert sorted(match) == list(range(n))


def test_deep_regime_torch_nlpd_matches_numpy_quadrature():
    # torch GH quadrature used sqrt(2)*x nodes inside delta = m + sqrt(2v)*x,
    # i.e. trained on delta ~ N(m, 2 v) while numpy scoring used N(m, v).
    # After the fix both marginalise the same N(m, v) posterior.
    torch = pytest.importorskip("torch")
    import quant_fund.models.deep_regime_mixture as drm

    rng = np.random.default_rng(0)
    n, h, r = 40, 2, 3
    out = {
        "delta_mean": torch.as_tensor(0.1 * rng.normal(size=(n, h)), dtype=torch.float32),
        "delta_var": torch.as_tensor(
            np.exp(-0.5 + 0.2 * rng.normal(size=(n, h))), dtype=torch.float32
        ),
        "mu": torch.as_tensor(rng.normal(size=(n, h)), dtype=torch.float32),
        "gate_logits": torch.as_tensor(rng.normal(size=(n, h, r)), dtype=torch.float32),
        "vres": torch.as_tensor(np.exp(-2.0 + 0.1 * rng.normal(size=(n, h))), dtype=torch.float32),
    }
    tau_val = np.exp(-1.0 + 0.1 * rng.normal(size=r)).astype(np.float32)

    class _NetStub:
        gate = "softmax"
        family = "gaussian"

        def tau(self):
            return torch.as_tensor(tau_val, dtype=torch.float32)

    y_std = torch.as_tensor(rng.normal(size=(n, h)), dtype=torch.float32)
    nodes_np, w_np = drm._gh_nodes(12)
    nodes_t = torch.as_tensor(nodes_np, dtype=torch.float32).view(1, 1, -1)
    logw_t = torch.as_tensor(np.log(w_np), dtype=torch.float32).view(1, 1, -1)
    floor = 0.05
    nlpd_t = float(
        drm._torch_nlpd(
            torch,
            _NetStub(),
            out,
            y_std,
            nodes_t=nodes_t,
            logw_t=logw_t,
            sigma_floor=floor,
        ).detach()
    )
    # numpy mirror on the same parameters (same standardised space)
    sigma2 = (
        tau_val.astype(float) ** 2
        + floor**2
        + np.asarray(out["vres"].numpy(), dtype=float)[:, :, None]
    )
    scales = np.broadcast_to(np.sqrt(sigma2), (n, h, r))
    logits = np.asarray(out["gate_logits"].numpy(), dtype=float)
    gates = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    gates = gates / gates.sum(axis=-1, keepdims=True)
    lp = drm.mixture_log_density(
        np.asarray(y_std.numpy(), dtype=float),
        np.asarray(out["mu"].numpy(), dtype=float),
        delta_mean=np.asarray(out["delta_mean"].numpy(), dtype=float),
        delta_var=np.asarray(out["delta_var"].numpy(), dtype=float),
        scales=scales,
        tails=np.full(r, 8.0),
        gates=gates,
        family="gaussian",
    )
    assert nlpd_t == pytest.approx(float(-np.mean(lp)), abs=2e-3)
    # and the buggy sqrt(2)*x convention is detectably different
    nodes_bad = torch.as_tensor(np.sqrt(2.0) * nodes_np, dtype=torch.float32).view(1, 1, -1)
    nlpd_bad = float(
        drm._torch_nlpd(
            torch,
            _NetStub(),
            out,
            y_std,
            nodes_t=nodes_bad,
            logw_t=logw_t,
            sigma_floor=floor,
        ).detach()
    )
    assert abs(nlpd_bad - nlpd_t) > 1e-3


def test_extra_tilt_source_density_integrates_to_mode_prob():
    # log_density used + log_ndtr (adding the truncation normalizer instead
    # of subtracting): the density then integrated to pi_z * ndtr(t)^2, not
    # pi_z, and score S_0 was not a proper negative log density.
    from quant_fund.models.extra_tilt_conformal import SourceConditional

    src = SourceConditional(
        gate=np.array([0.35, -0.2, 0.1]),
        mu_coef=np.array([[0.4, 0.3, -0.1], [-0.6, -0.2, 0.05]]),
        log_sigma_coef=np.array([-0.15, 0.2]),
    )
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 2))
    pi_pos = src.mode_prob(x)
    eps = 1e-6
    grid_pos = np.linspace(eps, 8.0, 60_001)
    grid_neg = np.linspace(-8.0, -eps, 60_001)
    mass_pos = np.empty(x.shape[0])
    mass_neg = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        xi = x[i : i + 1]
        xp = np.repeat(xi, grid_pos.size, axis=0)
        mass_pos[i] = np.trapezoid(np.exp(src.log_density(xp, grid_pos)), grid_pos)
        mass_neg[i] = np.trapezoid(np.exp(src.log_density(xp, grid_neg)), grid_neg)
    np.testing.assert_allclose(mass_pos, pi_pos, atol=2e-3)
    np.testing.assert_allclose(mass_neg, 1.0 - pi_pos, atol=2e-3)


def test_extra_tilt_source_mle_rejects_penalty(monkeypatch):
    import quant_fund.models.extra_tilt_conformal as mod

    monkeypatch.setattr(mod, "minimize", lambda *a, **k: _PenaltyResult(np.zeros(8)))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2))
    y = np.sign(rng.normal(size=80)) * np.exp(rng.normal(scale=0.5, size=80))
    with pytest.raises(ValueError, match="truncated-normal MLE"):
        mod.fit_source_model(x, y)


def test_gas_copula_fit_rejects_penalty(monkeypatch):
    import quant_fund.models.pair_vine_copula as mod

    monkeypatch.setattr(
        mod.opt, "minimize", lambda *a, **k: _PenaltyResult(np.asarray(a[1], dtype=float))
    )
    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    y = 0.7 * x + np.sqrt(1 - 0.49) * rng.normal(size=60)
    u = np.column_stack([x, y])
    out = mod.gas_copula_fit(u)
    # every start hit the 1e12 penalty -> nothing converged; honest zeros
    assert out["converged"] == 0.0
    assert out["loglik"] == 0.0


def test_dvine_edges_condition_on_interior_nodes():
    # tree-t edge (a, b) of a D-vine conditions on nodes a+1..b-1, not
    # on 0..t-1 — the old record mislabeled every edge with start > 0.
    from quant_fund.models.pair_vine_copula import _dvine_edges

    edges = _dvine_edges(5)
    assert edges[1] == [(0, 2, (1,)), (1, 3, (2,)), (2, 4, (3,))]
    assert edges[2] == [(0, 3, (1, 2)), (1, 4, (2, 3))]


def test_favar_forecast_rejects_nonfinite_history():
    from quant_fund.models.favar import favar_fit, favar_forecast, synth_favar

    d = synth_favar(t=120, seed=0)
    fit = favar_fit(d["y"], d["x"], r=2, p=1)
    y_bad = np.asarray(d["y"], dtype=float).copy()
    y_bad[5, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        favar_forecast(fit, y_bad, np.asarray(fit["factors"]), steps=2)


def test_rrc_taps_rejects_zero_beta():
    from quant_fund.models.rrc_filter import rrc_taps

    with pytest.raises(ValueError, match="beta"):
        rrc_taps(8, 33, beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        rrc_taps(8, 33, beta=-0.2)
    # control: normal call still works
    assert np.all(np.isfinite(rrc_taps(8, 33, beta=0.35)))
