"""Tests for quant_fund.models.sig_gan — SigCWGAN-lite time-series generation.

References (fetch-verified 2026-10): Liao, Ni, Szpruch, Wiese, Sabate-Vidales
& Xiao (2023, arXiv:2006.05421 — conditional Sig-W1 GAN; this module's
unconditional linear-witness variant); Chevyrev & Oberhauser (2022,
arXiv:1810.10971 — signature-moment MMD); Perez Arribas, Salvi & Szpruch
(2020, arXiv:2006.00218 — Sig-SDE expected-signature calibration).

All data here is SYNTHETIC (seeded GBM/OU fixtures) — algorithmic
correctness evidence, never market evidence; no Sharpe/P&L headline, no
live-trading claims. Torch tests skip cleanly when the nn extra is absent;
the numpy fallback and every metric are exercised torch-free.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys

import numpy as np
import pytest

from quant_fund.models import path_signatures as _sig
from quant_fund.models import sig_gan as sg


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch generator lane requires the nn extra (torch)"
)


def _ou_batch(seed: int = 0, n: int = 32, steps: int = 16) -> np.ndarray:
    return sg.sample_ou_paths(n, steps, 2, theta=1.5, mean=(0.3, -0.2), vol=0.25, seed=seed)


def _gbm_batch(seed: int = 0, n: int = 32, steps: int = 16) -> np.ndarray:
    return sg.sample_gbm_paths(n, steps, 2, drift=(0.05, -0.02), vol=(0.2, 0.3), rho=0.4, seed=seed)


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_torch_flag_is_consistent() -> None:
    assert sg.torch_backend_available() == _HAS_TORCH


def test_batch_signature_matches_canonical() -> None:
    """The internal batched mirror must equal path_signatures.signature."""
    paths = _ou_batch()
    aug = sg._time_augment(paths)
    batched = sg._signature_levels_batch(aug, 3)
    flat = np.concatenate([lvl.reshape(paths.shape[0], -1) for lvl in batched], axis=1)
    canonical = np.stack([_sig.signature(aug[i], 3) for i in range(aug.shape[0])])
    assert np.allclose(flat, canonical)


def test_expected_signature_shape_and_finite() -> None:
    paths = _ou_batch()
    es = sg.expected_signature(paths, order=2)
    d_aug = paths.shape[2] + 1
    assert es.shape == (d_aug + d_aug * d_aug,)
    assert np.all(np.isfinite(es))


def test_expected_signature_translation_invariant() -> None:
    paths = _ou_batch()
    base = sg.expected_signature(paths, order=2, time_augment=False)
    shifted = sg.expected_signature(paths + 5.0, order=2, time_augment=False)
    assert np.allclose(base, shifted)


def test_signature_distance_zero_for_identical() -> None:
    paths = _ou_batch()
    assert sg.signature_moment_distance(paths, paths, order=2) == pytest.approx(0.0, abs=1e-12)


def test_signature_distance_positive_for_different() -> None:
    a = sg.sample_ou_paths(32, 16, 1, theta=2.0, vol=0.1, seed=1)
    b = sg.sample_ou_paths(32, 16, 1, theta=0.05, vol=1.0, seed=2)
    assert sg.signature_moment_distance(a, b, order=2) > 0.0


def test_signature_distance_rejects_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="channel count"):
        sg.signature_moment_distance(_ou_batch(), _ou_batch()[:, :, :1])


def test_marginal_ks_zero_identical_positive_shifted() -> None:
    paths = _ou_batch()
    assert sg.marginal_ks_distance(paths, paths) == pytest.approx(0.0, abs=1e-12)
    shifted = sg.sample_ou_paths(32, 16, 2, mean=(3.0, -3.0), vol=0.5, seed=9)
    assert sg.marginal_ks_distance(paths, shifted) > 0.05


def test_acf_distance_zero_identical_positive_mean_reverting() -> None:
    paths = _ou_batch()
    assert sg.acf_distance(paths, paths) == pytest.approx(0.0, abs=1e-12)
    walk = sg.sample_gbm_paths(32, 16, 2, drift=0.0, vol=0.2, seed=5)
    assert sg.acf_distance(paths, np.log(walk)) > 0.0


def test_acf_distance_lag_bounds() -> None:
    paths = _ou_batch()
    with pytest.raises(ValueError, match="max_lag"):
        sg.acf_distance(paths, paths, max_lag=0)
    with pytest.raises(ValueError, match="max_lag"):
        sg.acf_distance(paths, paths, max_lag=100)


def test_metric_edges_fail_closed() -> None:
    with pytest.raises(ValueError, match="3-D"):
        sg.signature_moment_distance(np.zeros((4, 5)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="at least 2 paths"):
        sg.marginal_ks_distance(_ou_batch()[:1], _ou_batch()[:1])
    with pytest.raises(ValueError, match="finite"):
        bad = _ou_batch()
        bad[0, 0, 0] = np.nan
        sg.expected_signature(bad)
    with pytest.raises(ValueError, match="order"):
        sg.signature_moment_distance(_ou_batch(), _ou_batch(), order=0)
    with pytest.raises(ValueError, match="order"):
        sg.expected_signature(_ou_batch(), order=7)


def test_gbm_sampler_deterministic_and_positive() -> None:
    a = _gbm_batch(seed=7)
    b = _gbm_batch(seed=7)
    assert np.array_equal(a, b)
    assert np.all(a > 0.0)
    assert a.shape == (32, 17, 2)


def test_ou_sampler_reverts_to_mean() -> None:
    paths = sg.sample_ou_paths(64, 40, 1, theta=3.0, mean=0.5, vol=0.1, x0=-1.0, seed=3)
    assert float(paths[:, -1, 0].mean()) == pytest.approx(0.5, abs=0.15)


def test_samplers_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        sg.sample_ou_paths(0, 8, 1)
    with pytest.raises(ValueError, match="positive"):
        sg.sample_gbm_paths(4, 8, 1, vol=0.0)
    with pytest.raises(ValueError, match="rho"):
        sg.sample_ou_paths(4, 8, 2, rho=1.0)
    with pytest.raises(ValueError, match="s0"):
        sg.sample_gbm_paths(4, 8, 1, s0=-1.0)
    with pytest.raises(ValueError, match="vector"):
        sg.sample_ou_paths(4, 8, 2, theta=(1.0, 2.0, 3.0))


def test_fit_numpy_backend_reduces_signature_distance() -> None:
    real = _ou_batch(seed=11, n=48)
    cfg = sg.SigGANConfig(backend="numpy", iterations=150, batch_size=48, seed=11)
    res = sg.fit_sig_gan(real, config=cfg)
    assert res.backend == "numpy"
    assert res.paths.shape == real.shape
    assert res.sig_distance_after < res.sig_distance_before
    assert len(res.loss_trace) >= 1
    assert all(math.isfinite(v) for v in res.loss_trace)


def test_numpy_fallback_trace_is_nonincreasing() -> None:
    real = _ou_batch(seed=4, n=32)
    res = sg.fit_sig_gan(real, config=sg.SigGANConfig(backend="numpy", iterations=60, seed=4))
    tr = list(res.loss_trace)
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(tr))


def test_fit_determinism_same_seed() -> None:
    real = _ou_batch(seed=21, n=32)
    cfg = sg.SigGANConfig(iterations=60, batch_size=32, seed=21)
    r1 = sg.fit_sig_gan(real, config=cfg)
    r2 = sg.fit_sig_gan(real, config=cfg)
    assert np.array_equal(r1.paths, r2.paths)
    assert r1.loss_trace == r2.loss_trace


def test_sample_seeded_and_counted() -> None:
    real = _ou_batch(seed=8, n=24)
    res = sg.fit_sig_gan(real, config=sg.SigGANConfig(iterations=20, seed=8))
    a = res.sample(10, seed=1)
    b = res.sample(10, seed=1)
    assert a.shape == (10, 17, 2)
    assert np.array_equal(a, b)
    c = res.sample(10, seed=2)
    assert not np.array_equal(a, c)


def test_generate_paths_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="SigGANResult"):
        sg.generate_paths(object(), 4)


def test_fit_fail_closed_on_config() -> None:
    real = _ou_batch()
    with pytest.raises(ValueError, match="order"):
        sg.fit_sig_gan(real, config=sg.SigGANConfig(order=0))
    with pytest.raises(ValueError, match="backend"):
        sg.fit_sig_gan(real, config=sg.SigGANConfig(backend="tpu"))
    with pytest.raises(ValueError, match="iterations"):
        sg.fit_sig_gan(real, config=sg.SigGANConfig(iterations=0))
    with pytest.raises(ValueError, match="iterations"):
        sg.fit_sig_gan(real, config=sg.SigGANConfig(iterations=301))
    with pytest.raises(ValueError, match="lr"):
        sg.fit_sig_gan(real, config=sg.SigGANConfig(lr=0.0))
    with pytest.raises(ValueError, match="3-D"):
        sg.fit_sig_gan(np.zeros((4, 5)))


def test_module_imports_and_runs_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No top-level torch import; bench falls back to the numpy path."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_sg_no_torch", sg.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_sg_no_torch", probe)
    spec.loader.exec_module(probe)  # must import cleanly without torch
    assert probe.torch_backend_available() is False
    real = probe.sample_ou_paths(24, 12, 1, theta=1.5, vol=0.2, seed=0)
    res = probe.fit_sig_gan(
        real, config=probe.SigGANConfig(backend="auto", iterations=30, batch_size=24, seed=0)
    )
    assert res.backend == "numpy"
    assert np.all(np.isfinite(res.paths))
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.fit_sig_gan(real, config=probe.SigGANConfig(backend="torch", iterations=5))


def test_bench_returns_flat_synthetic_float_dict() -> None:
    out = sg.bench_sig_gan(seed=5)
    assert out, "bench must return a non-empty dict"
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert isinstance(val, float), (key, type(val))
        assert math.isfinite(val), key
    assert out["synthetic_torch_available"] == float(_HAS_TORCH)
    assert out["synthetic_determinism_delta"] == pytest.approx(0.0, abs=0.0)
    assert out["synthetic_ou_sig_distance_after"] <= out["synthetic_ou_sig_distance_before"] + 1e-9


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@requires_torch
def test_fit_torch_reduces_signature_distance() -> None:
    real = _ou_batch(seed=17, n=48)
    cfg = sg.SigGANConfig(backend="torch", iterations=150, batch_size=48, seed=17, patience=60)
    res = sg.fit_sig_gan(real, config=cfg)
    assert res.backend == "torch"
    assert res.paths.shape == real.shape
    assert res.sig_distance_after < res.sig_distance_before
    assert 1 <= res.iterations_run <= cfg.iterations
    assert len(res.loss_trace) == res.iterations_run
    assert res.n_parameters > 0


@requires_torch
def test_torch_fit_is_deterministic() -> None:
    real = _gbm_batch(seed=23, n=32)
    cfg = sg.SigGANConfig(backend="torch", iterations=40, batch_size=32, seed=23)
    r1 = sg.fit_sig_gan(real, config=cfg)
    r2 = sg.fit_sig_gan(real, config=cfg)
    assert np.array_equal(r1.paths, r2.paths)
    assert r1.loss_trace == r2.loss_trace


@requires_torch
def test_torch_early_stop_respects_budget() -> None:
    real = _ou_batch(seed=2, n=24)
    cfg = sg.SigGANConfig(backend="torch", iterations=200, patience=3, tol=1e9, seed=2)
    res = sg.fit_sig_gan(real, config=cfg)
    assert res.iterations_run <= cfg.patience + 1


@requires_torch
def test_torch_sample_count_and_seed() -> None:
    real = _ou_batch(seed=6, n=24)
    res = sg.fit_sig_gan(
        real, config=sg.SigGANConfig(backend="torch", iterations=15, batch_size=24, seed=6)
    )
    a = res.sample(8, seed=4)
    assert a.shape == (8, 17, 2)
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, res.sample(8, seed=4))


@requires_torch
def test_torch_gate_metrics_on_generated() -> None:
    real = _ou_batch(seed=31, n=48)
    res = sg.fit_sig_gan(
        real, config=sg.SigGANConfig(backend="torch", iterations=60, batch_size=48, seed=31)
    )
    gen = res.sample(48, seed=9)
    assert math.isfinite(sg.signature_moment_distance(real, gen))
    assert 0.0 <= sg.marginal_ks_distance(real, gen) <= 1.0
    assert sg.acf_distance(real, gen) >= 0.0
