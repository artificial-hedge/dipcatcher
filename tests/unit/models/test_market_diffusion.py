"""Joint DDPM correctness on labeled synthetic paths, never market evidence."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from quant_fund.models import market_diffusion as md
from quant_fund.models.market_diffusion import (
    DiffusionConfig,
    MarketDiffusion,
    cosine_betas,
    scenario_diagnostics,
    train_diffusion,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="joint DDPM training requires the nn extra"
)


def _fixture(n: int = 96) -> tuple[np.ndarray, list[list[datetime]], list[list[datetime]]]:
    rng = np.random.default_rng(51)
    common = rng.normal(size=(n, 3))
    common[:, 1:] += 0.4 * common[:, :-1]
    paths = np.stack((0.01 * common, 0.008 * common + 0.004 * rng.normal(size=(n, 3))), axis=-1)
    base = datetime(2020, 1, 1, tzinfo=UTC)
    events = [[base + timedelta(hours=4 * i + h) for h in range(3)] for i in range(n)]
    ready = [[t + timedelta(minutes=5) for t in row] for row in events]
    return paths, events, ready


def _kwargs(
    events: list[list[datetime]], ready: list[list[datetime]], n: int = 64
) -> dict[str, Any]:
    return dict(
        asset_ids=("A", "B"),
        timestamps=events,
        available_times=ready,
        cutoff=max(ready[n - 1]),
        data_source="seeded_joint_correctness",
        synthetic=True,
    )


def _config(**overrides: Any) -> DiffusionConfig:
    return replace(
        DiffusionConfig(n_steps=12, hidden_width=24, epochs=35, batch_size=32, seed=8), **overrides
    )


def test_cosine_schedule_matches_primary_formula_and_posterior() -> None:
    betas = cosine_betas(20)
    times = np.arange(21) / 20
    f = np.cos((times + 0.008) / 1.008 * np.pi / 2) ** 2
    np.testing.assert_allclose(betas, np.minimum(1 - f[1:] / f[:-1], 0.999), atol=1e-15)
    assert np.all((betas > 0) & (betas <= 0.999))
    alpha_bar = np.cumprod(1 - betas)
    previous = np.r_[1.0, alpha_bar[:-1]]
    posterior_variance = betas * (1 - previous) / (1 - alpha_bar)
    assert posterior_variance[0] == 0
    assert np.all(posterior_variance[1:] < betas[1:])
    assert alpha_bar[-1] < 1e-4


def test_module_is_importable_with_torch_blocked() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from quant_fund.models.market_diffusion import cosine_betas; "
            "assert len(cosine_betas(8)) == 8",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "changes",
    [
        {"n_steps": True},
        {"n_steps": 1},
        {"n_steps": 257},
        {"epochs": 0},
        {"epochs": 2_001},
        {"hidden_width": 3},
        {"hidden_width": 513},
        {"batch_size": 0},
        {"learning_rate": math.nan},
        {"learning_rate": 1},
        {"seed": -1},
        {"seed": 1.2},
        {"clip_denoised": "true"},
    ],
)
def test_invalid_configuration_fails_closed(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DiffusionConfig(**changes)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "univariate",
        "one_step",
        "nan",
        "infinity",
        "asset_count",
        "duplicate_ids",
        "blank_id",
        "naive_clock",
        "clock_shape",
        "reversed_clock",
        "ready_before_event",
        "duplicate_end",
        "naive_cutoff",
        "missing_source",
        "string_synthetic",
        "too_few_before_cutoff",
        "constant_asset",
        "wide_assets",
        "optimizer_budget",
    ],
)
def test_data_clock_identity_and_resource_contracts(case: str) -> None:
    paths, events, ready = _fixture()
    kwargs = _kwargs(events, ready)
    config = _config(epochs=1)
    if case == "empty":
        paths = paths[:0]
    elif case == "univariate":
        paths = paths[:, :, :1]
    elif case == "one_step":
        paths = paths[:, :1]
    elif case in {"nan", "infinity"}:
        paths[1, 1, 1] = np.nan if case == "nan" else np.inf
    elif case == "asset_count":
        kwargs["asset_ids"] = ("A",)
    elif case == "duplicate_ids":
        kwargs["asset_ids"] = ("A", "A")
    elif case == "blank_id":
        kwargs["asset_ids"] = ("A", " ")
    elif case == "naive_clock":
        events[0][0] = events[0][0].replace(tzinfo=None)
    elif case == "clock_shape":
        kwargs["timestamps"] = events[:-1]
    elif case == "reversed_clock":
        events[0] = list(reversed(events[0]))
    elif case == "ready_before_event":
        ready[0][0] = events[0][0] - timedelta(seconds=1)
    elif case == "duplicate_end":
        events[1] = list(events[0])
    elif case == "naive_cutoff":
        kwargs["cutoff"] = kwargs["cutoff"].replace(tzinfo=None)
    elif case == "missing_source":
        kwargs["data_source"] = ""
    elif case == "string_synthetic":
        kwargs["synthetic"] = "false"
    elif case == "too_few_before_cutoff":
        kwargs["cutoff"] = max(ready[4])
    elif case == "constant_asset":
        paths[:, :, 0] = 1
    elif case == "wide_assets":
        paths = np.zeros((8, 2, 129))
    elif case == "optimizer_budget":
        config = _config(batch_size=1, epochs=2_000)
    with pytest.raises(ValueError):
        MarketDiffusion(config).fit(paths, **kwargs)


@requires_torch
def test_train_prefix_is_independent_of_future_suffix_and_scaling() -> None:
    paths, events, ready = _fixture()
    changed = paths.copy()
    changed[64:] = 1000 + changed[64:] * 1e5
    kwargs = _kwargs(events, ready)
    first = train_diffusion(paths, config=_config(), **kwargs)
    second = train_diffusion(changed, config=_config(), **kwargs)
    prefix = train_diffusion(paths[:64], config=_config(), **_kwargs(events[:64], ready[:64]))
    assert (
        first.fit_info is not None and second.fit_info is not None and prefix.fit_info is not None
    )
    assert first.fit_info.n_training_paths == 64
    assert first.fit_info.n_excluded_paths == 32
    assert first.fit_info.training_data_sha256 == second.fit_info.training_data_sha256
    assert (
        first.fit_info.model_sha256 == second.fit_info.model_sha256 == prefix.fit_info.model_sha256
    )
    np.testing.assert_array_equal(
        first.sample_scenarios(32).paths, second.sample_scenarios(32).paths
    )
    np.testing.assert_array_equal(
        first.sample_scenarios(32).paths, prefix.sample_scenarios(32).paths
    )


@requires_torch
def test_available_time_not_event_time_selects_training_windows() -> None:
    paths, events, ready = _fixture()
    kwargs = _kwargs(events, ready)
    ready[20][1] = kwargs["cutoff"] + timedelta(days=10)
    model = train_diffusion(paths, config=_config(epochs=2), **kwargs)
    assert model.fit_info is not None
    assert model.fit_info.n_training_paths == 63
    assert model.fit_info.max_training_available_time <= kwargs["cutoff"].isoformat()


@requires_torch
def test_actual_joint_training_sampling_and_honesty_without_torch_at_inference(
    monkeypatch: Any,
) -> None:
    paths, events, ready = _fixture(128)
    model = train_diffusion(paths, config=_config(epochs=70), **_kwargs(events, ready, 128))
    info = model.fit_info
    assert info is not None
    assert len(info.epoch_losses) == 70 and np.isfinite(info.epoch_losses).all()
    assert info.training_probe_final_loss < info.training_probe_initial_loss
    noisy = np.zeros((1, 3, 2))
    perturbed = noisy.copy()
    perturbed[0, 0, 0] = 0.5
    # Changing one asset/horizon influences other coordinates of the learned
    # joint denoiser; this does not assert successful learned market dependence.
    delta = model.predict_noise(perturbed, 4) - model.predict_noise(noisy, 4)
    assert np.any(np.abs(delta[0, :, 1]) > 1e-8)
    monkeypatch.setattr(md, "_torch", lambda: pytest.fail("numpy inference imported torch"))
    batch = model.sample_scenarios(256, seed=19)
    assert batch.paths.shape == (256, 3, 2) and np.isfinite(batch.paths).all()
    assert batch.paths.flags.writeable is False
    assert batch.asset_ids == ("A", "B")
    assert batch.synthetic and batch.label == "SYNTHETIC" and not batch.market_evidence
    assert not batch.live_pnl_claim and batch.model_sha256 == info.model_sha256
    covariance = np.cov(batch.paths.reshape(256, -1), rowvar=False)
    assert covariance.shape == (6, 6) and np.linalg.eigvalsh(covariance).min() > -1e-15
    assert np.all(covariance.diagonal() > 0)
    assert np.all(batch.paths >= paths.min(axis=0) - 1e-15)
    assert np.all(batch.paths <= paths.max(axis=0) + 1e-15)
    np.testing.assert_array_equal(batch.paths, model.sample_scenarios(256, seed=19).paths)
    assert not np.array_equal(batch.paths, model.sample_scenarios(256, seed=20).paths)


@requires_torch
def test_empirical_training_still_produces_synthetic_scenarios() -> None:
    paths, events, ready = _fixture()
    kwargs = _kwargs(events, ready)
    kwargs.update(data_source="declared_empirical_fixture_only", synthetic=False)
    model = train_diffusion(paths, config=_config(epochs=2), **kwargs)
    batch = model.sample_scenarios(4)
    assert batch.training_synthetic is False
    assert batch.synthetic is True and batch.market_evidence is False
    # The fixture remains synthetic test input; this case tests flag inheritance,
    # not the existence of an empirical dataset.


@requires_torch
def test_global_torch_rng_and_controls_are_restored() -> None:
    import torch

    paths, events, ready = _fixture()
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    train_diffusion(paths, config=_config(epochs=2), **_kwargs(events, ready))
    assert torch.equal(state, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()


@requires_torch
def test_failed_refit_with_nonfinite_training_loss_does_not_serve_stale_model(
    monkeypatch: Any,
) -> None:
    paths, events, ready = _fixture()
    kwargs = _kwargs(events, ready)
    model = train_diffusion(paths, config=_config(epochs=2), **kwargs)
    original = md._finite_loss

    def nonfinite(_: float) -> float:
        return original(math.nan)

    monkeypatch.setattr(md, "_finite_loss", nonfinite)
    with pytest.raises(FloatingPointError, match="non-finite DDPM training loss"):
        model.fit(paths, **kwargs)
    assert model.fit_info is None
    with pytest.raises(RuntimeError, match="successfully fitted"):
        model.sample_scenarios(4)


@requires_torch
def test_diagnostics_use_only_postcutoff_windows_and_never_issue_pass() -> None:
    paths, events, ready = _fixture()
    model = train_diffusion(paths, config=_config(epochs=2), **_kwargs(events, ready))
    kwargs = dict(
        asset_ids=("A", "B"),
        timestamps=events[64:],
        available_times=ready[64:],
        data_source="synthetic_holdout",
        synthetic=True,
        n_scenarios=32,
        seed=31,
    )
    report = model.diagnostics(paths[64:], **kwargs)
    assert report["status"] == "descriptive_distribution_gaps"
    assert np.asarray(report["covariance_abs_gap"]).shape == (2, 2)
    assert np.asarray(report["tail_quantile_abs_gap"]).shape == (4, 2)
    assert len(report["lag1_correlation_abs_gap_by_asset"]) == 2
    assert report["generated_label"] == "SYNTHETIC" and report["sota_established"] is False
    assert "passed" not in report
    kwargs.update(timestamps=events[:32], available_times=ready[:32])
    with pytest.raises(ValueError, match="entirely after"):
        model.diagnostics(paths[:32], **kwargs)


def test_diagnostic_identity_and_undefined_autocorrelation() -> None:
    paths, _, _ = _fixture()
    same = scenario_diagnostics(paths, paths)
    assert np.max(same["covariance_abs_gap"]) == 0
    assert np.max(same["tail_quantile_abs_gap"]) == 0
    constant = np.ones_like(paths)
    result = scenario_diagnostics(constant, constant)
    assert result["lag1_correlation_abs_gap_by_asset"] == [None, None]


@requires_torch
def test_sampler_and_model_binding_reject_invalid_or_changed_contract() -> None:
    paths, events, ready = _fixture()
    model = train_diffusion(paths, config=_config(epochs=2), **_kwargs(events, ready))
    for bad in (0, True, -2, 100_001):
        with pytest.raises(ValueError):
            model.sample_scenarios(bad)
    with pytest.raises(ValueError, match="resource budget"):
        model.sample_scenarios(100_000)
    with pytest.raises(ValueError):
        model.predict_noise(np.zeros((1, 4, 2)), 1)
    model.config = _config(n_steps=14)
    with pytest.raises(RuntimeError, match="config changed"):
        model.sample_scenarios(4)


@requires_torch
def test_changed_parameters_and_provenance_cannot_keep_old_scenario_identity() -> None:
    from dataclasses import replace

    paths, events, ready = _fixture()
    model = train_diffusion(paths, config=_config(epochs=2), **_kwargs(events, ready))
    info = model.fit_info
    assert info is not None
    model.fit_info = replace(info, data_source="changed-source")
    with pytest.raises(RuntimeError, match="provenance changed"):
        model.sample_scenarios(4)
    model.fit_info = info
    assert model._params is not None
    altered = model._params.mean.copy()
    altered[0] += 1.0
    model._params = replace(model._params, mean=altered)
    with pytest.raises(RuntimeError, match="parameters changed"):
        model.sample_scenarios(4)
