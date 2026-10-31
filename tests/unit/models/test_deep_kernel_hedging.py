"""Tests for quant_fund.models.deep_kernel_hedging — Dupret, Hainaut & Motte (2026).

References: Dupret, J.-L., Hainaut, D. & Motte, E. (2026), "Deep kernel
hedging", arXiv:2609.34474, https://doi.org/10.48550/arXiv.2609.34474
(RKHS hedging functional with a learned neural kernel; generalized
representer theorem, Thm 3.1; RFF approximation with uniform tail bound,
Prop 3.2; primal RFF equivalence, Prop 2.1 / Lemma 3.2; Algorithms 1 & 2);
Wilson, Hu, Salakhutdinov & Xing (2016, deep kernel learning,
arXiv:1511.02222); Rahimi & Recht (2007, random features); Schölkopf,
Herbrich & Smola (2001, generalized representer theorem); Rockafellar &
Uryasev (2000, CVaR representation); Buehler, Gonon, Teichmann & Wood
(2019, deep hedging baseline and simulators, arXiv:1802.03042); Chen
(1958) / Chevyrev & Kormilitzin (2016/2025) for signature features.

All data here is SYNTHETIC (seeded GBM and regime-switch paths from
``deep_hedging``'s simulators) — correctness evidence for the algorithm,
never market evidence; no live-trading claims. Torch tests skip cleanly
when the nn extra is absent; the numpy lane (signature features, hedging
Gram, representer solve, RFF solve, exact/RFF kernel hedges) runs without
torch.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from quant_fund.models import deep_hedging as dh
from quant_fund.models import deep_kernel_hedging as dkh
from quant_fund.models.path_signatures import signature


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="deep kernel hedging training requires the nn extra (torch)"
)

T_MAT = 0.5
STEPS = 12
DT = T_MAT / STEPS
# Trained DKH configuration for the SYNTHETIC validation tests (tuned once on
# seeded synthetic paths; deterministic full-batch CPU training).
DKH_CFG = {"lam": 3e-4, "n_rff": 400, "gamma0": 4.0, "epochs": 100, "lr": 3e-3}


def _gbm(n: int, seed: int, s0: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    paths = dh.simulate_gbm_paths(n, STEPS, s0=s0, sigma=0.2, dt=DT, seed=seed)
    return paths, dh.european_payoff(paths, s0)


def _rs(n: int, seed: int, s0: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    paths = dh.simulate_regime_switch_paths(
        n, STEPS, s0=s0, sigma_low=0.1, sigma_high=0.4, dt=DT, seed=seed
    )
    return paths, dh.european_payoff(paths, s0)


# ---------------------------------------------------------------------------
# numpy lane: signature features (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_signature_features_shapes_determinism_and_zero_start() -> None:
    paths, _ = _gbm(5, 1)
    f2 = dkh.signature_features(paths, 2)
    f2b = dkh.signature_features(paths, 2)
    assert f2.shape == (5, STEPS, 2**3 - 2)
    assert dkh.signature_features(paths, 3).shape == (5, STEPS, 2**4 - 2)
    assert np.array_equal(f2, f2b)
    # k = 0: the prefix is a single point, all levels >= 1 vanish.
    assert np.all(f2[:, 0] == 0.0)
    assert np.all(np.isfinite(f2))


def test_signature_features_match_public_signature_and_reference_mode() -> None:
    """Incremental Chen accumulation == reference mode == public signature()."""
    paths, _ = _gbm(3, 2)
    inc = dkh.signature_features(paths, 3)
    ref = dkh.signature_features(paths, 3, incremental=False)
    assert np.allclose(inc, ref, atol=1e-12)
    for order in (1, 2, 3):
        feats = dkh.signature_features(paths, order)
        for i in range(paths.shape[0]):
            t_frac = np.arange(paths.shape[1]) / STEPS
            log_rel = np.log(paths[i] / paths[i, 0])
            pts = np.column_stack([t_frac, log_rel])
            for k in range(1, STEPS):
                assert np.allclose(feats[i, k], signature(pts[: k + 1], order), atol=1e-12)


def test_signature_features_level1_and_shuffle_identities() -> None:
    """Level 1 is (t_k, log S_k/S_0); level 2 obeys the shuffle identities."""
    paths, _ = _gbm(4, 3)
    feats = dkh.signature_features(paths, 2)
    t = np.arange(STEPS + 1) / STEPS
    log_rel = np.log(paths / paths[:, :1])
    for k in range(1, STEPS):
        assert np.allclose(feats[:, k, 0], t[k], atol=1e-12)
        assert np.allclose(feats[:, k, 1], log_rel[:, k], atol=1e-12)
        # Sig^11 = t^2/2, Sig^22 = x^2/2, Sig^12 + Sig^21 = t*x (shuffle).
        assert np.allclose(feats[:, k, 2], t[k] ** 2 / 2.0, atol=1e-12)
        assert np.allclose(feats[:, k, 5], log_rel[:, k] ** 2 / 2.0, atol=1e-12)
        assert np.allclose(feats[:, k, 3] + feats[:, k, 4], t[k] * log_rel[:, k], atol=1e-12)


def test_signature_features_are_adapted() -> None:
    """Features at step k depend only on prices up to time t_k (no lookahead)."""
    paths, _ = _gbm(4, 4)
    k0 = 6
    perturbed = paths.copy()
    perturbed[:, k0 + 1 :] *= 1.05
    f_orig = dkh.signature_features(paths, 3)
    f_pert = dkh.signature_features(perturbed, 3)
    assert np.allclose(f_orig[:, : k0 + 1], f_pert[:, : k0 + 1], atol=1e-12)
    assert not np.allclose(f_orig[:, k0 + 1 :], f_pert[:, k0 + 1 :])


def test_signature_features_fail_closed() -> None:
    with pytest.raises(ValueError, match="2-D"):
        dkh.signature_features(np.ones(5), 2)
    with pytest.raises(ValueError, match="at least two time points"):
        dkh.signature_features(np.ones((3, 1)), 2)
    with pytest.raises(ValueError, match="strictly positive"):
        dkh.signature_features(-np.ones((3, 4)), 2)
    with pytest.raises(ValueError, match="finite"):
        dkh.signature_features(np.full((3, 4), np.nan), 2)
    for bad in (0, 7, -1, True):
        with pytest.raises(ValueError, match="order must be an integer"):
            dkh.signature_features(np.ones((3, 4)), bad)


def test_standardize_features_and_fail_closed() -> None:
    paths, _ = _gbm(6, 5)
    feats = dkh.signature_features(paths, 2)
    z, mu, sd = dkh.standardize_features(feats)
    assert z.shape == feats.shape
    assert np.allclose(mu, feats.mean(axis=(0, 1)), atol=1e-12)
    assert np.allclose(z.mean(axis=(0, 1)), 0.0, atol=1e-12)
    # Applying the stored statistics reproduces the same transform.
    z2, _, _ = dkh.standardize_features(feats, mu, sd)
    assert np.allclose(z, z2, atol=1e-12)
    with pytest.raises(ValueError, match="3-D"):
        dkh.standardize_features(feats[0])
    with pytest.raises(ValueError, match="length d="):
        dkh.standardize_features(feats, np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="strictly positive"):
        dkh.standardize_features(feats, mu, np.zeros_like(sd))


# ---------------------------------------------------------------------------
# numpy lane: RFF construction and the representer/primal solves
# ---------------------------------------------------------------------------


def test_rff_draw_seeded_determinism() -> None:
    W1, b1 = dkh.rff_draw(4, 50, seed=11)
    W2, b2 = dkh.rff_draw(4, 50, seed=11)
    W3, b3 = dkh.rff_draw(4, 50, seed=12)
    assert W1.shape == (50, 4) and b1.shape == (50,)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2)
    assert not np.array_equal(W1, W3)
    assert np.all((b1 >= 0.0) & (b1 <= 2.0 * math.pi))
    with pytest.raises(ValueError, match="p_dim"):
        dkh.rff_draw(0, 10, seed=0)
    with pytest.raises(ValueError, match="n_features"):
        dkh.rff_draw(2, 0, seed=0)


def test_rff_features_approximate_exact_rbf_kernel() -> None:
    """Documented approximation error: unbiased RFF, sup-err shrinking in D.

    P(sup |K~_D - K| > eps) <= C eps^-2 exp(-c D eps^2) (Dupret et al. 2026,
    Prop 3.2; Rahimi & Recht 2007): the max error must decay like O(D^-1/2).
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((40, 3))
    exact = dkh.rbf_kernel_matrix(z, z, 1.0)
    errs = {}
    for d_feat in (100, 10000):
        W, b = dkh.rff_draw(3, d_feat, seed=0)
        y = dkh.rff_features(z, W, b)
        approx = y @ y.T
        errs[d_feat] = float(np.max(np.abs(approx - exact)))
        if d_feat == 10000:
            # Diagonal: the base kernel is normalized, K(z, z) = 1.
            assert np.max(np.abs(np.diag(approx) - 1.0)) < 0.05
    assert errs[10000] < errs[100]
    assert errs[10000] < 0.05
    with pytest.raises(ValueError, match="latent dim p"):
        W, b = dkh.rff_draw(3, 10, seed=0)
        dkh.rff_features(rng.standard_normal((5, 2)), W, b)


def test_rff_features_fail_closed() -> None:
    W, b = dkh.rff_draw(2, 5, seed=0)
    with pytest.raises(ValueError, match="b must have length"):
        dkh.rff_features(np.ones((3, 2)), W, b[:4])
    with pytest.raises(ValueError, match="finite"):
        dkh.rff_features(np.full((3, 2), np.inf), W, b)
    with pytest.raises(ValueError, match="gamma must be positive"):
        dkh.rbf_kernel_matrix(np.ones((2, 2)), np.ones((3, 2)), 0.0)
    with pytest.raises(ValueError, match="latent dims must match"):
        dkh.rbf_kernel_matrix(np.ones((2, 2)), np.ones((3, 5)), 1.0)


def test_hedging_gram_small_case_and_symmetry() -> None:
    """Q_ij = sum_{k,l} G_i[k] G_j[l] K(x_ik, x_jl) against an explicit loop."""
    rng = np.random.default_rng(7)
    n_paths, n_steps = 3, 2
    latent = rng.standard_normal((n_paths * n_steps, 4))
    gains = rng.standard_normal((n_paths, n_steps))
    K = dkh.rbf_kernel_matrix(latent, latent, 0.7)
    Q = dkh.hedging_gram(K, gains)
    expected = np.empty((n_paths, n_paths))
    for i in range(n_paths):
        for j in range(n_paths):
            s = 0.0
            for k in range(n_steps):
                for ell in range(n_steps):
                    s += gains[i, k] * gains[j, ell] * K[i * n_steps + k, j * n_steps + ell]
            expected[i, j] = s
    assert np.allclose(Q, expected, atol=1e-12)
    assert np.allclose(Q, Q.T, atol=0.0)
    with pytest.raises(ValueError, match=r"\(n_paths\*n_steps\)"):
        dkh.hedging_gram(K, rng.standard_normal((n_paths, n_steps + 1)))


def test_hedging_gram_is_psd() -> None:
    """Q is a Gram matrix of RKHS sections => positive semidefinite."""
    rng = np.random.default_rng(8)
    n_paths, n_steps = 6, 3
    latent = rng.standard_normal((n_paths * n_steps, 3))
    gains = rng.standard_normal((n_paths, n_steps))
    Q = dkh.hedging_gram(dkh.rbf_kernel_matrix(latent, latent, 1.0), gains)
    eigs = np.linalg.eigvalsh(Q)
    assert np.all(eigs >= -1e-10)


def test_solve_representer_alpha_small_system() -> None:
    """alpha* = (Q + N lam I)^{-1} H — the paper's square-loss closed form."""
    Q = np.array([[2.0, 1.0], [1.0, 2.0]])
    alpha = dkh.solve_representer_alpha(Q, np.array([1.0, 0.0]), 0.5)
    # system = Q + 2*0.5*I = [[3, 1], [1, 3]]; inverse = (1/8)[[3, -1], [-1, 3]]
    assert np.allclose(alpha, np.array([3.0 / 8.0, -1.0 / 8.0]), atol=1e-12)
    with pytest.raises(ValueError, match="lam must be positive"):
        dkh.solve_representer_alpha(Q, np.array([1.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="target must have length"):
        dkh.solve_representer_alpha(Q, np.array([1.0]), 0.5)


def test_rff_primal_equivalence_lemma() -> None:
    """Lemma 3.2 / Prop 2.1: dual alpha on Q~=ZZ' equals the primal beta solve."""
    rng = np.random.default_rng(9)
    n_paths, d_feat = 12, 20
    Z = rng.standard_normal((n_paths, d_feat))
    target = rng.standard_normal(n_paths)
    lam = 1e-2
    alpha = dkh.solve_representer_alpha(Z @ Z.T, target, lam)
    beta = dkh.solve_rff_beta(Z, target, lam)
    # Identical fitted values Z Z' alpha == Z beta on the training paths...
    assert np.allclose(Z @ (Z.T @ alpha), Z @ beta, atol=1e-8)
    # ...and identical objective values (1/N)||H - fit||^2 + lam * norm^2.
    obj_dual = np.mean((target - Z @ (Z.T @ alpha)) ** 2) + lam * alpha @ (Z @ Z.T) @ alpha
    obj_primal = np.mean((target - Z @ beta) ** 2) + lam * beta @ beta
    assert abs(obj_dual - obj_primal) < 1e-10
    with pytest.raises(ValueError, match="target must have length"):
        dkh.solve_rff_beta(Z, target[:-1], lam)


def test_rff_aggregate_shape_and_fail_closed() -> None:
    rng = np.random.default_rng(10)
    y = rng.standard_normal((4, 3, 7))
    gains = rng.standard_normal((4, 3))
    Z = dkh.rff_aggregate(y, gains)
    assert Z.shape == (4, 7)
    assert np.allclose(Z, np.einsum("ikd,ik->id", y, gains))
    with pytest.raises(ValueError, match="gains must have shape"):
        dkh.rff_aggregate(y, rng.standard_normal((4, 2)))


# ---------------------------------------------------------------------------
# numpy lane: exact representer hedge and RFF hedge (no torch)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gbm_small() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded SYNTHETIC GBM train/eval batches at repo scale (s0 = 100)."""
    tr, pay = _gbm(64, 7, s0=100.0)
    ev, pay_ev = _gbm(512, 99, s0=100.0)
    return tr, pay, ev, pay_ev


def test_hedged_error_matches_deep_hedging_loss(gbm_small) -> None:
    """E_i(phi; v0=0, B=1) == deep_hedging's per-path loss at cost_rate=0."""
    tr, pay, _, _ = gbm_small
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((tr.shape[0], STEPS)) * 0.1
    err = dkh.hedged_error(tr, pos, pay)
    assert np.allclose(err, dh.hedged_loss(tr, pos, pay, cost_rate=0.0), atol=1e-12)
    with pytest.raises(ValueError, match="positions must have shape"):
        dkh.hedged_error(tr, pos[:, :-1], pay)
    with pytest.raises(ValueError, match="payoff must be"):
        dkh.hedged_error(tr, pos, pay[:-1])


def test_exact_kernel_hedge_reduces_risk_and_is_deterministic(gbm_small) -> None:
    """SYNTHETIC: representer hedge beats unhedged out-of-sample; fit is stable."""
    tr, pay, ev, pay_ev = gbm_small
    fit = dkh.exact_kernel_hedge(tr, pay, gamma=0.5, lam=1e-6, order=2)
    pos_ev = fit.hedge_positions(ev)
    err = dkh.hedged_error(ev, pos_ev, pay_ev)
    assert pos_ev.shape == (ev.shape[0], STEPS)
    assert np.all(np.isfinite(pos_ev))
    assert err.var() < pay_ev.var()  # hedged risk <= unhedged risk
    assert err.var() < 0.75 * pay_ev.var()
    # hedge_positions on the training paths reproduces the stored positions.
    assert np.allclose(fit.hedge_positions(tr), fit.train_positions, atol=1e-10)
    # Deterministic: an identical refit gives identical coefficients.
    fit2 = dkh.exact_kernel_hedge(tr, pay, gamma=0.5, lam=1e-6, order=2)
    assert np.array_equal(fit.alpha, fit2.alpha)
    with pytest.raises(ValueError, match="gamma must be positive"):
        dkh.exact_kernel_hedge(tr, pay, gamma=0.0, lam=1e-6, order=2)
    with pytest.raises(ValueError, match="lam must be positive"):
        dkh.exact_kernel_hedge(tr, pay, gamma=0.5, lam=-1.0, order=2)
    with pytest.raises(ValueError, match="n_steps"):
        fit.hedge_positions(tr[:, :-1])


def test_rff_kernel_hedge_converges_to_exact(gbm_small) -> None:
    """SYNTHETIC: RFF primal hedge -> exact representer hedge as D grows.

    Dupret et al. (2026, Prop 3.2 / Thm 3.3): the RFF Gram converges in
    operator norm and the RFF hedging problems converge to the exact one.
    """
    tr, pay, ev, pay_ev = gbm_small
    fit = dkh.exact_kernel_hedge(tr, pay, gamma=0.5, lam=1e-6, order=2)
    var_exact = dkh.hedged_error(ev, fit.hedge_positions(ev), pay_ev).var()
    var_small = dkh.hedged_error(
        ev,
        dkh.rff_kernel_hedge(
            tr, pay, gamma=0.5, lam=1e-6, n_rff=100, seed=3, order=2
        ).hedge_positions(ev),
        pay_ev,
    ).var()
    var_big = dkh.hedged_error(
        ev,
        dkh.rff_kernel_hedge(
            tr, pay, gamma=0.5, lam=1e-6, n_rff=1600, seed=3, order=2
        ).hedge_positions(ev),
        pay_ev,
    ).var()
    assert abs(var_big - var_exact) < abs(var_small - var_exact)
    assert var_big < pay_ev.var()  # hedged risk <= unhedged at large D
    # Seeded determinism of the RFF draw.
    a = dkh.rff_kernel_hedge(tr, pay, gamma=0.5, lam=1e-6, n_rff=100, seed=3, order=2)
    b = dkh.rff_kernel_hedge(tr, pay, gamma=0.5, lam=1e-6, n_rff=100, seed=3, order=2)
    assert np.array_equal(a.beta, b.beta)
    with pytest.raises(ValueError, match="n_rff"):
        dkh.rff_kernel_hedge(tr, pay, gamma=0.5, lam=1e-6, n_rff=0, seed=3, order=2)


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module has no top-level torch import; torch entry points fail closed."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_dkh_no_torch", dkh.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_dkh_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core stays usable while torch is blocked
    paths, pay = _gbm(8, 0, s0=100.0)
    fit = probe.exact_kernel_hedge(paths, pay, gamma=1.0, lam=1e-3, order=2)
    assert fit.hedge_positions(paths).shape == (8, STEPS)
    assert probe.rff_kernel_hedge(paths, pay, gamma=1.0, lam=1e-3, n_rff=16, seed=0, order=1)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.deep_kernel_hedge(paths, pay, epochs=1)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rs_low_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded SYNTHETIC regime-switch train/eval batches (independent seeds).

    The two-state regime is latent, so the optimal hedge is path-dependent —
    the setting where the paper's signature features and kernel inductive
    bias matter (Dupret et al. 2026, §1: low-data advantage; §4.2).
    """
    tr, pay = _rs(250, 7)
    ev, pay_ev = _rs(1024, 99)
    return tr, pay, ev, pay_ev


@requires_torch
def test_deep_kernel_hedge_deterministic_and_seeded() -> None:
    tr, pay = _rs(64, 7)
    a = dkh.deep_kernel_hedge(tr, pay, n_rff=50, order=2, epochs=20, lr=3e-3, seed=5)
    b = dkh.deep_kernel_hedge(tr, pay, n_rff=50, order=2, epochs=20, lr=3e-3, seed=5)
    c = dkh.deep_kernel_hedge(tr, pay, n_rff=50, order=2, epochs=20, lr=3e-3, seed=6)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.beta, b.beta)
    assert not np.array_equal(a.positions, c.positions)
    assert a.positions.shape == (64, STEPS)
    assert np.all(np.isfinite(a.positions))
    assert len(a.loss_curve) == 20
    assert np.all(np.isfinite(a.loss_curve))
    assert a.objective <= a.loss_curve[0] + 1e-12
    assert a.gamma > 0.0


@requires_torch
def test_deep_kernel_hedge_fail_closed() -> None:
    tr, pay = _gbm(16, 0)
    with pytest.raises(ValueError, match="unknown loss kind"):
        dkh.deep_kernel_hedge(tr, pay, loss="entropic", epochs=1)
    with pytest.raises(ValueError, match="alpha_cvar"):
        dkh.deep_kernel_hedge(tr, pay, loss="cvar", alpha_cvar=1.5, epochs=1)
    with pytest.raises(ValueError, match="lam must be positive"):
        dkh.deep_kernel_hedge(tr, pay, lam=0.0, epochs=1)
    with pytest.raises(ValueError, match="gamma0 must be positive"):
        dkh.deep_kernel_hedge(tr, pay, gamma0=-1.0, epochs=1)
    with pytest.raises(ValueError, match="epochs must be"):
        dkh.deep_kernel_hedge(tr, pay, epochs=0)
    with pytest.raises(ValueError, match="lr must be positive"):
        dkh.deep_kernel_hedge(tr, pay, lr=0.0, epochs=1)
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        dkh.deep_kernel_hedge(tr, pay, weight_decay=-0.1, epochs=1)
    with pytest.raises(ValueError, match="n_rff"):
        dkh.deep_kernel_hedge(tr, pay, n_rff=0, epochs=1)
    with pytest.raises(ValueError, match="p_dim"):
        dkh.deep_kernel_hedge(tr, pay, p_dim=0, epochs=1)
    with pytest.raises(ValueError, match="hidden must be"):
        dkh.deep_kernel_hedge(tr, pay, hidden=(), epochs=1)
    with pytest.raises(ValueError, match="order must be"):
        dkh.deep_kernel_hedge(tr, pay, order=7, epochs=1)
    with pytest.raises(ValueError, match="payoff must be"):
        dkh.deep_kernel_hedge(tr, pay[:-1], epochs=1)


@requires_torch
def test_strategy_positions_shape_and_fail_closed(rs_low_data) -> None:
    tr, pay, _, _ = rs_low_data
    res = dkh.deep_kernel_hedge(tr, pay, order=2, seed=7, **DKH_CFG)
    pos = res.strategy_positions(tr[:8])
    assert pos.shape == (8, STEPS)
    assert np.all(np.isfinite(pos))
    with pytest.raises(ValueError, match="n_steps"):
        res.strategy_positions(tr[:, :-1])


@requires_torch
def test_low_data_advantage_regime_switch_synthetic(rs_low_data) -> None:
    """SYNTHETIC headline: DKH beats plain deep hedging in the low-data regime.

    N = 250 latent-regime training paths (the paper's smallest N), 1024
    independently seeded evaluation paths, quadratic hedging (variance of
    the terminal hedging error, cost_rate = 0), the metric of the paper's
    Table 4.1. Correctness evidence on simulated paths, never market
    evidence; no Sharpe/P&L headline (AGENTS.md honesty contract).
    """
    tr, pay, ev, pay_ev = rs_low_data
    dkh_res = dkh.deep_kernel_hedge(tr, pay, order=2, seed=7, **DKH_CFG)
    dh_res = dh.deep_hedge(tr, pay, cost_rate=0.0, risk="variance", epochs=200, seed=7)
    err_dkh = dh.hedged_loss(ev, dkh_res.strategy_positions(ev), pay_ev, cost_rate=0.0)
    err_dh = dh.hedged_loss(ev, dh_res.strategy_positions(ev), pay_ev, cost_rate=0.0)
    var_dkh, var_dh, var_un = err_dkh.var(), err_dh.var(), pay_ev.var()
    assert var_dkh < var_dh  # deep kernel hedge at least as good as deep hedge
    assert var_dkh <= 0.95 * var_dh  # with a non-trivial margin at this seed
    assert var_dh < var_un  # both hedged risks <= unhedged
    assert var_dkh < var_un


@requires_torch
def test_gbm_deep_kernel_hedge_beats_unhedged_under_friction() -> None:
    """SYNTHETIC GBM: DKH trained frictionless still dominates unhedged with
    proportional costs applied at evaluation (Davis & Norman frictions via
    deep_hedging's accounting; frictions are outside the paper's model)."""
    tr, pay = _gbm(250, 7)
    ev, pay_ev = _gbm(1024, 99)
    res = dkh.deep_kernel_hedge(tr, pay, order=1, seed=7, **DKH_CFG)
    pos = res.strategy_positions(ev)
    zeros = np.zeros_like(pos)
    for cost in (1e-4, 1e-3):
        comp = dh.hedged_pnl_components(ev, pos, pay_ev, cost_rate=cost)
        comp_un = dh.hedged_pnl_components(ev, zeros, pay_ev, cost_rate=cost)
        var = dh.risk_measure(comp["loss"], "variance")
        var_un = dh.risk_measure(comp_un["loss"], "variance")
        es = dh.risk_measure(comp["loss"], "expected_shortfall", alpha=0.9)
        es_un = dh.risk_measure(comp_un["loss"], "expected_shortfall", alpha=0.9)
        assert var < var_un  # hedged risk <= unhedged under friction
        assert es < es_un


@requires_torch
def test_cvar_deep_kernel_hedge_reduces_tail_risk_synthetic() -> None:
    """SYNTHETIC CVaR hedging (paper Alg. 2): trained Rockafellar-Uryasev
    objective yields lower tail risk than unhedged on fresh paths."""
    tr, pay = _rs(250, 7)
    ev, pay_ev = _rs(512, 99)
    res = dkh.deep_kernel_hedge(tr, pay, loss="cvar", alpha_cvar=0.1, order=2, seed=7, **DKH_CFG)
    assert res.loss_kind == "cvar"
    assert np.isfinite(res.zeta)
    err = dh.hedged_loss(ev, res.strategy_positions(ev), pay_ev, cost_rate=0.0)
    es = dh.risk_measure(err, "expected_shortfall", alpha=0.9)
    es_un = dh.risk_measure(pay_ev, "expected_shortfall", alpha=0.9)
    assert es < es_un  # hedged tail risk <= unhedged
    assert err.var() < pay_ev.var()
