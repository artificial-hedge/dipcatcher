"""Tests for quant_fund.models.gslice — G-SLiCE (Berndt et al. 2026).

References: Berndt, Farjallah, Seute, Saqur, Walker & Stühmer (2026),
"Universal Time Series Generation with Neural Controlled Differential
Equations", arXiv:2605.28507; Lipman et al. (2023, flow matching); Tong et
al. (2023, OT-CFM); Kollovieh et al. (2023, TSFlow grid version); Gneiting
& Raftery (2007, energy score); Rasmussen & Williams (2006, GP regression);
Lyons (1998) / Chevyrev & Kormilitzin (2016, signature features).

All data here is SYNTHETIC (GP priors, regime-switching increments) —
correctness evidence for the algorithm, never market evidence; no
live-trading claims. Torch tests skip cleanly when the nn extra is absent.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from quant_fund.models import gslice as gs


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="G-SLiCE training requires the nn extra (torch)"
)

GRID = np.linspace(0.0, 1.0, 17)
N_TRAIN = 128
N_EVAL = 64


def _law(seed: int, n: int = N_TRAIN) -> np.ndarray:
    """SYNTHETIC regime-switching path law (non-Gaussian increments)."""
    return gs.synthetic_switching_paths(n, GRID, seed=seed)


def _cfg(**kw: object) -> gs.GSliceConfig:
    base: dict[str, object] = dict(
        hidden_dim=16, n_blocks=2, block_size=16, epochs=15, lr=3e-3, batch_size=64
    )
    base.update(kw)
    return gs.GSliceConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# numpy layer (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_gp_kernel_matrix_rbf_exact_values() -> None:
    t = np.array([0.0, 0.5, 1.0])
    k = gs.gp_kernel_matrix(t, kernel="rbf", length_scale=0.5, amplitude=2.0)
    assert k.shape == (3, 3)
    assert np.allclose(np.diag(k), 4.0)  # k(t, t) = a²
    assert np.isclose(k[0, 2], 4.0 * np.exp(-0.5 * (1.0 / 0.5) ** 2))
    assert np.allclose(k, k.T)


def test_gp_kernel_matrix_matern32_and_wiener_exact() -> None:
    t = np.array([0.0, 0.25, 1.0])
    m = gs.gp_kernel_matrix(t, kernel="matern32", length_scale=0.5, amplitude=1.5)
    r = np.sqrt(3.0) * 0.25 / 0.5
    assert np.isclose(m[0, 1], 1.5**2 * (1.0 + r) * np.exp(-r))
    w = gs.gp_kernel_matrix(t, kernel="wiener", amplitude=1.0)
    assert np.all(w[0] == 0.0) and w[2, 2] == 1.0
    assert w[1, 2] == 0.25  # min(t1, t2) anchored at t0 = 0


def test_gp_kernel_matrix_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown kernel"):
        gs.gp_kernel_matrix(GRID, kernel="cauchy")
    with pytest.raises(ValueError, match="strictly increasing"):
        gs.gp_kernel_matrix(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="length_scale"):
        gs.gp_kernel_matrix(GRID, length_scale=0.0)
    with pytest.raises(ValueError, match="amplitude"):
        gs.gp_kernel_matrix(GRID, amplitude=-1.0)
    with pytest.raises(ValueError, match="at least 2"):
        gs.gp_kernel_matrix(np.array([0.5]))


def test_sample_gp_paths_shape_moments_and_determinism() -> None:
    a = gs.sample_gp_paths(256, GRID, kernel="rbf", amplitude=0.8, rng=11)
    b = gs.sample_gp_paths(256, GRID, kernel="rbf", amplitude=0.8, rng=11)
    c = gs.sample_gp_paths(256, GRID, kernel="rbf", amplitude=0.8, rng=12)
    assert a.shape == (256, 17, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    # GP moments: centered mean, marginal variance ~ a² (Monte-Carlo tol).
    assert abs(float(a.mean())) < 0.15
    assert abs(float(a[:, 8, 0].var()) - 0.64) < 0.2
    assert np.all(np.isfinite(a))


def test_sample_gp_paths_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        gs.sample_gp_paths(0, GRID)
    with pytest.raises(ValueError, match="mean"):
        gs.sample_gp_paths(4, GRID, mean=np.zeros(9))
    with pytest.raises(ValueError, match="finite"):
        gs.sample_gp_paths(4, GRID, mean=np.full(17, np.nan))


def test_gp_posterior_interpolates_and_shrinks_variance() -> None:
    t_obs = np.array([0.0, 0.5])
    y_obs = np.array([[0.3], [-0.2]])
    mean, cov = gs.gp_posterior_moments(
        GRID, t_obs, y_obs, kernel="rbf", length_scale=0.4, noise=1e-8
    )
    assert mean.shape == (17, 1) and cov.shape == (17, 17)
    # Near-noiseless conditioning: posterior mean hits the observations.
    assert np.isclose(mean[0, 0], 0.3, atol=1e-4)
    assert np.isclose(mean[8, 0], -0.2, atol=1e-4)
    prior_cov = gs.gp_kernel_matrix(GRID, kernel="rbf", length_scale=0.4)
    assert np.all(np.diag(cov) <= np.diag(prior_cov) + 1e-9)


def test_gp_posterior_fail_closed() -> None:
    with pytest.raises(ValueError, match="obs_times must be non-empty"):
        gs.gp_posterior_moments(GRID, np.array([]), np.empty((0, 1)))
    with pytest.raises(ValueError, match="obs_values"):
        gs.gp_posterior_moments(GRID, np.array([0.0, 0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="noise"):
        gs.gp_posterior_moments(GRID, np.array([0.0]), np.array([[0.0]]), noise=-1.0)
    with pytest.raises(ValueError, match="finite"):
        gs.gp_posterior_moments(GRID, np.array([0.0, 0.5]), np.array([[0.0], [np.nan]]))


def test_sample_gp_posterior_pins_observations() -> None:
    t_obs = np.array([0.0, 0.5])
    y_obs = np.array([[0.5], [-0.4]])
    mean, cov = gs.gp_posterior_moments(
        GRID, t_obs, y_obs, kernel="rbf", length_scale=0.4, noise=1e-8
    )
    draws = gs.sample_gp_posterior_paths(128, mean, cov, rng=5)
    assert draws.shape == (128, 17, 1)
    # Draws concentrate on the observations far tighter than the prior sd.
    assert float(draws[:, 0, 0].std()) < 0.05
    assert abs(float(draws[:, 0, 0].mean()) - 0.5) < 0.05
    a = gs.sample_gp_posterior_paths(8, mean, cov, rng=5)
    b = gs.sample_gp_posterior_paths(8, mean, cov, rng=5)
    assert np.array_equal(a, b)


def test_context_logsignature_known_value() -> None:
    # Straight line: logsignature is exactly its level-1 increment (higher
    # Lyndon coordinates vanish) — Lyons/Chevyrev-Kormilitzin.
    path = np.array([[[0.0], [0.2], [0.4]]])  # (1, 3, 1), uniform in value
    feats = gs.context_logsignature(path, order=2)
    assert feats.shape == (1, 3)  # (t, x) augmented: l1=2, l2=1
    assert np.isclose(feats[0, 1], 0.4)  # level-1 x-coordinate = total dx
    assert abs(feats[0, 2]) < 1e-12  # straight line has no area


def test_context_logsignature_invariances_and_fail_closed() -> None:
    p = _law(3, n=4)[:, :5]
    a = gs.context_logsignature(p, order=2)
    b = gs.context_logsignature(p + 10.0, order=2)  # translation-invariant
    assert np.allclose(a, b)
    with pytest.raises(ValueError, match="order"):
        gs.context_logsignature(p, order=5)
    with pytest.raises(ValueError, match="ndim|3-D"):
        gs.context_logsignature(np.zeros((4, 5)))


def test_build_context_broadcast_and_fail_closed() -> None:
    data = _law(1, n=6)
    ctx = gs.build_context(GRID[:5], data[:, :5], order=2)
    assert ctx.obs_times.shape == (6, 5)
    assert ctx.obs_values.shape == (6, 5, 1)
    assert ctx.features.shape == (6, 3)
    t_per = np.tile(GRID[:5], (6, 1))
    ctx2 = gs.build_context(t_per, data[:, :5], order=2)
    assert np.array_equal(ctx.features, ctx2.features)
    with pytest.raises(ValueError, match="at least 2"):
        gs.build_context(GRID[:1], data[:, :1])
    with pytest.raises(ValueError, match="obs_times"):
        gs.build_context(np.array([0.5, 0.1, 0.0, 0.0, 0.0]), data[:, :5])
    with pytest.raises(ValueError, match="length m"):
        gs.build_context(np.array([0.0, 0.1]), data[:, :5])


def test_ot_coupling_optimal_assignment() -> None:
    # x1 is a permuted copy of x0; OT must recover the identity pairing.
    x0 = _law(5, n=8)
    perm_truth = np.array([3, 0, 7, 1, 6, 2, 5, 4])
    x1 = x0[perm_truth]
    perm = gs.ot_coupling(x0, x1)
    inv = np.empty(8, dtype=int)
    inv[perm_truth] = np.arange(8)
    assert np.array_equal(perm, inv)
    # Coupled cost <= any fixed permutation's cost (here: identity).
    cost_c = float(np.sum((x0 - x1[perm]) ** 2))
    cost_id = float(np.sum((x0 - x1) ** 2))
    assert cost_c <= cost_id
    with pytest.raises(ValueError, match="share shape"):
        gs.ot_coupling(x0, x1[:4])


def test_interpolant_endpoints_and_fail_closed() -> None:
    x0 = _law(7, n=4)
    x1 = _law(8, n=4)
    assert np.array_equal(gs.interpolant(x0, x1, 0.0), x0)
    assert np.array_equal(gs.interpolant(x0, x1, 1.0), x1)
    mid = gs.interpolant(x0, x1, 0.5)
    assert np.allclose(mid, 0.5 * (x0 + x1))
    per = gs.interpolant(x0, x1, np.linspace(0.0, 1.0, 4))
    assert np.allclose(per[0], x0[0]) and np.allclose(per[3], x1[3])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gs.interpolant(x0, x1, -0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gs.interpolant(x0, x1, 1.1)
    with pytest.raises(ValueError, match="share shape"):
        gs.interpolant(x0, x1[:2], 0.5)


def test_path_energy_score_proper_ordering() -> None:
    # Properness sanity: an ensemble from the true law scores better than a
    # shifted ensemble on held-out paths of the same law.
    held = _law(20, n=32)
    good = _law(21, n=64)
    bad = _law(21, n=64) + 1.5
    es_good = gs.path_energy_score(good, held)
    es_bad = gs.path_energy_score(bad, held)
    assert es_good < es_bad
    with pytest.raises(ValueError, match=">= 2"):
        gs.path_energy_score(good[:1], held)
    with pytest.raises(ValueError, match="share"):
        gs.path_energy_score(good[:, :, :], held[:, :, :1] * np.ones((32, 17, 2)))


def test_path_band_coverage_nominal_and_fail_closed() -> None:
    # iid ensemble + held-out from the same law: marginal coverage of the
    # central band approaches the nominal level (Monte-Carlo band).
    ens = _law(30, n=400)
    obs = _law(31, n=400)
    band = gs.path_band_coverage(ens, obs, level=0.8)
    assert 0.7 < band["coverage"] < 0.9
    assert band["mean_width"] > 0.0
    with pytest.raises(ValueError, match="level"):
        gs.path_band_coverage(ens, obs, level=1.5)
    with pytest.raises(ValueError, match=">= 2"):
        gs.path_band_coverage(ens[:1], obs)


def test_path_pit_values_open_interval_and_centering() -> None:
    ens = _law(40, n=200)
    obs = _law(41, n=100)
    pits = gs.path_pit_values(ens, obs)
    assert pits.shape == obs.shape
    assert np.all(pits > 0.0) and np.all(pits < 1.0)
    # Under the same law PITs are centered near 0.5.
    assert abs(float(pits.mean()) - 0.5) < 0.05
    # A shifted ensemble pushes PITs to the boundary region.
    pits_shift = gs.path_pit_values(ens + 3.0, obs)
    assert float(pits_shift.mean()) < 0.1


def test_evaluate_samples_scorecard_keys() -> None:
    gen = _law(50, n=64)
    obs = _law(51, n=32)
    out = gs.evaluate_samples(gen, obs)
    for key in (
        "gslice_energy_score_mean",
        "gslice_coverage_50",
        "gslice_coverage_80",
        "gslice_coverage_95",
        "gslice_width_80",
        "gslice_pit_mean",
        "gslice_pit_sd",
        "gslice_pit_ks_pvalue",
    ):
        assert key in out and np.isfinite(out[key])
    with pytest.raises(ValueError, match="levels"):
        gs.evaluate_samples(gen, obs, levels=())


def test_synthetic_laws_determinism_and_structure() -> None:
    a = gs.synthetic_switching_paths(32, GRID, seed=2)
    b = gs.synthetic_switching_paths(32, GRID, seed=2)
    c = gs.synthetic_switching_paths(32, GRID, seed=3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.all(a[:, 0, 0] == 0.0)
    # Two vol regimes visible: per-step |increment| dispersion is bimodal-ish.
    inc = np.abs(np.diff(a[:, :, 0], axis=1)).reshape(-1)
    assert inc.std() > 0.0
    d = gs.synthetic_gp_paths(16, GRID, drift=0.7, seed=4)
    slope = float(np.polyfit(GRID, d[:, :, 0].mean(axis=0), 1)[0])
    assert slope > 0.5  # drift recovered on average
    with pytest.raises(ValueError, match="p_stay"):
        gs.synthetic_switching_paths(4, GRID, p_stay=1.5)


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module has no top-level torch import; torch entry points fail closed."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_gs_no_torch", gs.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_gs_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core stays usable while torch is blocked
    paths = probe.sample_gp_paths(4, GRID, rng=0)
    assert paths.shape == (4, 17, 1)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.train_gslice(paths)
    model = probe.GSliceModel(
        net=None,
        grid=GRID,
        config=probe.GSliceConfig(),
        d_x=1,
        d_ctx=0,
        d_aux=0,
        loss_curve=np.zeros(1),
        train_n=4,
    )
    with pytest.raises(ImportError, match=r"'nn' extra"):
        model.sample(2)


def test_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="divisible"):
        gs.GSliceConfig(hidden_dim=16, block_size=6)
    with pytest.raises(ValueError, match="unknown kernel"):
        gs.GSliceConfig(kernel="periodic")
    with pytest.raises(ValueError, match="sig_order"):
        gs.GSliceConfig(sig_order=5)
    with pytest.raises(ValueError, match="epochs"):
        gs.GSliceConfig(epochs=0)
    with pytest.raises(ValueError, match="lr"):
        gs.GSliceConfig(lr=0.0)
    with pytest.raises(ValueError, match="n_blocks"):
        gs.GSliceConfig(n_blocks=0)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@requires_torch
def test_expm_scaled_matches_scipy_expm() -> None:
    """The exact-flow transition equals scipy's expm up to ~1e-6."""
    import torch
    from scipy.linalg import expm

    torch.manual_seed(0)
    mats = torch.randn(32, 4, 4, dtype=torch.float64) * 0.3
    ours = gs._expm_scaled(torch, mats).numpy()
    ref = np.stack([expm(m) for m in mats.numpy()])
    assert np.allclose(ours, ref, atol=1e-6)
    # near-identity for near-zero generators
    assert np.allclose(
        gs._expm_scaled(torch, torch.zeros(2, 4, 4, dtype=torch.float64)).numpy(),
        np.broadcast_to(np.eye(4), (2, 4, 4)),
    )


@requires_torch
def test_slice_field_exact_reduction_zero_transitions() -> None:
    """A^i = 0 -> Phi = I -> hidden stays h0 -> velocity is time-constant.

    Exact reduction: with all transition weights zeroed, the SLiCE
    recurrence is the identity, so each block's readout path is constant
    along the grid and the field output equals block-1's constant velocity.
    """
    import torch

    torch.manual_seed(0)
    net = gs._build_field(torch, _cfg(), d_x=1, d_ctx=0, d_aux=0)
    for block in net.blocks:
        block.weight.data.zero_()
    x = torch.randn(4, 17, 1)
    s = torch.full((4,), 0.3)
    v = net(x, s, torch.zeros(4, 0), torch.zeros(4, 17, 0))
    assert v.shape == (4, 17, 1)
    assert torch.allclose(v, v[:, :1, :].expand(4, 17, 1))


@requires_torch
def test_train_loss_decreases_and_is_finite() -> None:
    model = gs.train_gslice(_law(0), grid=GRID, config=_cfg(epochs=15))
    assert np.all(np.isfinite(model.loss_curve))
    assert model.loss_curve.shape == (15,)
    assert model.loss_curve[-1] < model.loss_curve[0]


@requires_torch
def test_trained_ensemble_beats_prior_energy_score() -> None:
    """Trained G-SLiCE samples score better than the raw GP prior (proper ES)."""
    model = gs.train_gslice(_law(0), grid=GRID, config=_cfg(epochs=25))
    held = _law(99, n=N_EVAL)
    gen = model.sample(N_EVAL, seed=1)
    es_trained = gs.path_energy_score(gen, held)
    es_prior = gs.path_energy_score(gs.sample_gp_paths(N_EVAL, GRID, rng=7), held)
    assert es_trained < es_prior


@requires_torch
def test_drifted_law_decisive_score_gain_and_coverage() -> None:
    """On a mean-shifted GP law the prior cannot transport the drift:
    trained ES must beat prior ES clearly, and coverage sits near nominal."""
    train = gs.synthetic_gp_paths(
        N_TRAIN,
        GRID,
        kernel="matern32",
        length_scale=0.5,
        amplitude=0.8,
        drift=0.6,
        seed=0,
    )
    held = gs.synthetic_gp_paths(
        N_EVAL,
        GRID,
        kernel="matern32",
        length_scale=0.5,
        amplitude=0.8,
        drift=0.6,
        seed=9,
    )
    model = gs.train_gslice(train, grid=GRID, config=_cfg(epochs=25))
    gen = model.sample(N_EVAL, seed=1)
    ev = gs.evaluate_samples(gen, held)
    pr = gs.evaluate_samples(gs.sample_gp_paths(N_EVAL, GRID, rng=7), held)
    gain = pr["gslice_energy_score_mean"] - ev["gslice_energy_score_mean"]
    assert gain > 0.05 * pr["gslice_energy_score_mean"]
    assert 0.55 < ev["gslice_coverage_80"] < 0.95
    assert ev["gslice_coverage_95"] > ev["gslice_coverage_50"]


@requires_torch
def test_train_and_sample_determinism() -> None:
    a = gs.train_gslice(_law(0), grid=GRID, config=_cfg(epochs=8, seed=5))
    b = gs.train_gslice(_law(0), grid=GRID, config=_cfg(epochs=8, seed=5))
    c = gs.train_gslice(_law(0), grid=GRID, config=_cfg(epochs=8, seed=6))
    assert np.array_equal(a.loss_curve, b.loss_curve)
    assert not np.array_equal(a.loss_curve, c.loss_curve)
    sa = a.sample(8, seed=2)
    sb = a.sample(8, seed=2)
    assert np.array_equal(sa, sb)


@requires_torch
def test_sampler_methods_and_fail_closed() -> None:
    model = gs.train_gslice(_law(0), grid=GRID, config=_cfg(epochs=8))
    e = model.sample(8, seed=1, method="euler")
    m = model.sample(8, seed=1, method="midpoint")
    assert e.shape == m.shape == (8, 17, 1)
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(m))
    assert not np.array_equal(e, m)  # different integrators, both finite
    with pytest.raises(ValueError, match="method"):
        model.sample(4, method="rk4")
    with pytest.raises(ValueError, match="flow_steps"):
        model.sample(4, flow_steps=0)
    with pytest.raises(ValueError, match="n_paths"):
        model.sample(0)


@requires_torch
def test_conditional_generation_pins_prefix() -> None:
    """GP-posterior prior: conditional samples concentrate near observed values."""
    data = _law(0)
    ctx = gs.build_context(GRID[:5], data[:, :5], order=2)
    model = gs.train_gslice(data, grid=GRID, context=ctx, config=_cfg(epochs=15))
    held = _law(7, n=1)
    gen = model.sample_conditional(48, GRID[:5], held[:, :5], seed=3)
    assert gen.shape == (48, 17, 1)
    # At the observed start the ensemble is pinned near the observation,
    # far tighter than the marginal data-law spread at that grid point.
    assert abs(float(gen[:, 0, 0].mean()) - held[0, 0, 0]) < 0.1
    assert float(gen[:, 0, 0].std()) < 0.1


@requires_torch
def test_signature_context_changes_output_and_is_required() -> None:
    data = _law(0)
    ctx = gs.build_context(GRID[:5], data[:, :5], order=2)
    model = gs.train_gslice(data, grid=GRID, context=ctx, config=_cfg(epochs=10))
    held = _law(7, n=2)
    c1 = gs.build_context(GRID[:5], held[0:1, :5], order=2)
    c2 = gs.build_context(GRID[:5], held[1:2, :5], order=2)
    s1 = model.sample(16, context=c1, seed=2)
    s2 = model.sample(16, context=c2, seed=2)
    assert not np.array_equal(s1, s2)  # signature context modulates the field
    with pytest.raises(ValueError, match="context"):
        model.sample(4)  # trained with context -> required
    with pytest.raises(ValueError, match="n_context"):
        model.sample(4, context=gs.build_context(GRID[:5], held[:, :5]))


@requires_torch
def test_train_fail_closed_edges() -> None:
    data = _law(0)
    with pytest.raises(ValueError, match="3-D"):
        gs.train_gslice(data[:, :, 0])
    with pytest.raises(ValueError, match="grid length"):
        gs.train_gslice(data, grid=np.linspace(0, 1, 9))
    ctx_bad = gs.build_context(GRID[:5], data[:, :5])
    ctx_bad = gs.GSliceContext(
        obs_times=ctx_bad.obs_times[:4],
        obs_values=ctx_bad.obs_values[:4],
        features=ctx_bad.features[:4],
    )
    with pytest.raises(ValueError, match="one row per data path"):
        gs.train_gslice(data, grid=GRID, context=ctx_bad)
    with pytest.raises(ValueError, match="data_paths"):
        gs.train_gslice(data[:3])


@requires_torch
def test_bench_gslice_scorecard() -> None:
    out = gs.bench_gslice(n_train=128, n_eval=48, config=gs.GSliceConfig(epochs=30, batch_size=64))
    assert "gslice_synth_energy_score_mean" in out
    assert "synthetic_gslice_synth_energy_score_prior" in out
    assert "gslice_synth_coverage_80" in out
    assert out["synthetic_gslice_synth_energy_score_gain"] > 0.0
    assert out["synthetic_gslice_synth_loss_drop"] > 0.0
    assert all(np.isfinite(v) for v in out.values())
