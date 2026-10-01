"""SYNTHETIC window, score, train-baseline and immutable receipt checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scripts.eval_blueprint_diffusion import (
    MODEL_NAMES,
    EvaluationConfig,
    baseline_ensembles,
    complete_run,
    make_windows,
    reserve_run,
    score_ensembles,
    verify_run,
)
from scripts.eval_blueprint_graph import PhasePanel, canonical_hash

from quant_fund.metrics.energy_score import energy_score


def _phase(n=15, indices=None):
    calendar = tuple(datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n + 1))
    if indices is None:
        indices = list(range(n))
    origins = tuple(calendar[i] for i in indices)
    endpoints = np.asarray([[calendar[i + 1]] * 2 for i in indices], dtype=object)
    available = endpoints.copy()
    features_ready = np.asarray([[t] * 2 for t in origins], dtype=object)
    values = np.asarray([[i / 100, -i / 200] for i in indices])
    phase = PhasePanel(np.ones((len(indices), 2, 4)), values, origins, features_ready, available)
    return phase, endpoints, calendar


def _training():
    rng = np.random.default_rng(44)
    values = rng.normal(size=(20, 5, 3)) / 100
    values[..., 1] = values[..., 0] * 0.8 + values[..., 1] * 0.2
    return values


def _ensemble(training=None, n=128):
    training = _training() if training is None else training
    ensembles, _ = baseline_ensembles(training, n=n)
    # Tests use a copy of the empirical bootstrap as DDPM placeholder;
    # this is arithmetic verification, not a generator performance claim.
    ensembles["ddpm"] = ensembles["block_bootstrap"].copy()
    return ensembles


def _reserved(tmp_path):
    output = tmp_path / "run"
    source = tmp_path / "code.py"
    source.write_text("# synthetic source fixture\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = {"config": "synthetic fixed protocol"}
    reserve_run(output, protocol, {"script": source.read_bytes()})
    payload = {
        "protocol_sha256": canonical_hash(protocol),
        "code_files": {"script": {"path": str(source), "sha256": digest}},
        "source_files": {},
        "synthetic": True,
    }
    return output, source, payload


def test_fixed_protocol_is_small_and_has_no_score_threshold():
    config = EvaluationConfig()
    assert (config.asset_count, config.horizon, config.ensemble_size) == (8, 5, 128)
    assert (config.n_steps, config.hidden_dim, config.epochs, config.batch_size) == (
        16,
        32,
        100,
        64,
    )
    assert config.seed == 7 and config.clip_denoised is True


def test_nonoverlapping_paths_keep_original_event_and_release_clocks():
    phase, endpoints, calendar = _phase(n=17)
    releases = phase.target_available_times.copy()
    releases[:, 1] = [t + timedelta(hours=1) for t in releases[:, 1]]
    window = make_windows(
        replace(phase, target_available_times=releases), endpoints, calendar, asset_count=2
    )
    assert window.paths.shape == (3, 5, 2)
    np.testing.assert_array_equal(window.paths.reshape(-1, 2), phase.targets[:15])
    assert window.origin_times[0][0] == calendar[0]
    assert window.event_times[0][0] == calendar[1]
    assert window.available_times[0][0] == calendar[1] + timedelta(hours=1)
    assert window.event_times[0][-1] < window.event_times[1][0]
    assert window.audit["discarded_return_rows"] == 2


def test_complete_case_gaps_split_windows_and_discard_each_remainder():
    phase, endpoints, calendar = _phase(n=25, indices=list(range(7)) + list(range(9, 22)))
    window = make_windows(phase, endpoints, calendar, asset_count=2)
    assert window.audit["consecutive_segments"] == 2
    assert window.audit["discarded_return_rows"] == 5
    assert window.paths[:, 0, 0].tolist() == [0, 0.09, 0.14]
    assert window.event_times[1][0] == calendar[10]


@pytest.mark.parametrize(
    "fault",
    [
        "naive",
        "not_next",
        "unequal",
        "release_before_event",
        "future_feature",
        "calendar_duplicate",
        "origin_duplicate",
        "nonfinite",
        "bad_shape",
    ],
)
def test_invalid_source_clocks_or_windows_fail_closed(fault):
    phase, endpoints, calendar = _phase()
    if fault == "naive":
        endpoints[0, 0] = endpoints[0, 0].replace(tzinfo=None)
    elif fault == "not_next":
        endpoints[0, :] = calendar[2]
    elif fault == "unequal":
        endpoints[0, 1] = calendar[2]
    elif fault == "release_before_event":
        ready = phase.target_available_times.copy()
        ready[0, 0] = calendar[0]
        phase = replace(phase, target_available_times=ready)
    elif fault == "future_feature":
        ready = phase.feature_available_times.copy()
        ready[0, 0] = calendar[1]
        phase = replace(phase, feature_available_times=ready)
    elif fault == "calendar_duplicate":
        calendar = (calendar[0],) + calendar
    elif fault == "origin_duplicate":
        phase = replace(phase, timestamps=(phase.timestamps[0],) + phase.timestamps[:-1])
    elif fault == "nonfinite":
        phase.targets[0, 0] = np.nan
    elif fault == "bad_shape":
        endpoints = endpoints[:, :1]
    with pytest.raises(ValueError):
        make_windows(phase, endpoints, calendar, asset_count=2)


def test_future_suffix_changes_do_not_change_training_windows():
    phase, endpoints, calendar = _phase(n=20)
    original = make_windows(phase, endpoints, calendar, asset_count=2)
    values = phase.targets.copy()
    values[15:] *= 10000
    changed = make_windows(replace(phase, targets=values), endpoints, calendar, asset_count=2)
    np.testing.assert_array_equal(original.paths[:3], changed.paths[:3])
    assert original.event_times[:3] == changed.event_times[:3]


def test_fewer_than_two_full_paths_fail():
    phase, endpoints, calendar = _phase(n=9)
    with pytest.raises(ValueError, match="at least two"):
        make_windows(phase, endpoints, calendar, asset_count=2)


def test_baselines_are_reproducible_train_only_and_shape_matched():
    training = _training()
    first, parameters = baseline_ensembles(training)
    second, _ = baseline_ensembles(training.copy())
    for name in MODEL_NAMES[1:]:
        assert first[name].shape == (128, 5, 3)
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_allclose(parameters["gaussian_mean"], training.reshape(20, -1).mean(axis=0))
    np.testing.assert_allclose(
        parameters["gaussian_covariance"], np.cov(training.reshape(20, -1), rowvar=False, ddof=1)
    )
    for draw in first["block_bootstrap"]:
        assert any(np.array_equal(draw, path) for path in training)
    for a in range(3):
        assert set(first["iid_marginal_empirical"][..., a].ravel()) <= set(training[..., a].ravel())
    # Preserved dependence in the fitted joint Gaussian is explicit;
    # no threshold on finite random draws is used as market evidence.
    assert parameters["gaussian_covariance"][0, 1] > 0


@pytest.mark.parametrize("shape", [(7, 5, 3), (20, 3), (20, 5, 3)])
def test_bad_baseline_inputs_rejected(shape):
    values = np.zeros(shape)
    if shape == (20, 5, 3):
        values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        baseline_ensembles(values)


def test_energy_score_known_values_and_independent_pairwise_arithmetic():
    assert energy_score(np.asarray([[0.0, 0], [0, 0]]), np.asarray([3.0, 4])) == 5
    assert energy_score(np.asarray([[0.0, 0], [2, 0]]), np.asarray([1.0, 0])) == 0
    rng = np.random.default_rng(6)
    ensemble, observation = rng.normal(size=(9, 4)), rng.normal(size=4)
    first = sum(float(np.linalg.norm(row - observation)) for row in ensemble) / 9
    second = sum(
        float(np.linalg.norm(ensemble[i] - ensemble[j]))
        for i in range(9)
        for j in range(9)
        if i != j
    ) / (2 * 9 * 8)
    assert energy_score(ensemble, observation) == pytest.approx(first - second, abs=1e-14)


def test_score_same_frozen_ensemble_and_all_adverse_scores_retained():
    reference = _training()
    ensembles = _ensemble()
    # Deliberately adverse DDPM placeholder stays in the report.
    ensembles["ddpm"] = ensembles["ddpm"] + 1
    report, losses = score_ensembles(ensembles, reference)
    assert report["mean_ddpm_minus_baseline"]["block_bootstrap"] > 0
    assert report["acceptance_threshold"] is None
    for name in MODEL_NAMES:
        expected = [
            energy_score(ensembles[name].reshape(128, -1), row.ravel()) for row in reference
        ]
        np.testing.assert_allclose(losses[name], expected, rtol=0, atol=0)
        assert report["models"][name]["diagnostics"]["synthetic_generated"] is True
    assert reference.shape == (20, 5, 3)


@pytest.mark.parametrize("fault", ["missing", "unequal_count", "unequal_shape", "nonfinite"])
def test_score_mismatches_fail(fault):
    ensembles = _ensemble()
    if fault == "missing":
        del ensembles["ddpm"]
    elif fault == "unequal_count":
        ensembles["ddpm"] = ensembles["ddpm"][:5]
    elif fault == "unequal_shape":
        ensembles["ddpm"] = ensembles["ddpm"][:, :4]
    elif fault == "nonfinite":
        ensembles["ddpm"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        score_ensembles(ensembles, _training())


def test_receipt_is_exclusive_and_verifies_exact_artifacts(tmp_path):
    output, _, payload = _reserved(tmp_path)
    receipt = complete_run(output, payload, {"arrays.npz": b"synthetic arrays bytes"})
    verified = verify_run(output)
    assert receipt.is_file() and verified["synthetic"] is True
    assert (
        verified["artifacts"]["arrays.npz"] == hashlib.sha256(b"synthetic arrays bytes").hexdigest()
    )
    with pytest.raises(FileExistsError):
        reserve_run(output, {}, {})
    with pytest.raises(FileExistsError):
        complete_run(output, payload, {"arrays.npz": b"replacement"})
    assert (output / "arrays.npz").read_bytes() == b"synthetic arrays bytes"


def test_interrupted_reserved_attempt_cannot_be_restarted(tmp_path):
    output, _, _ = _reserved(tmp_path)
    with pytest.raises(FileExistsError):
        reserve_run(output, {}, {})
    assert not (output / "receipt.json").exists()


def test_protocol_drift_before_completion_cannot_receive_a_receipt(tmp_path):
    output, _, payload = _reserved(tmp_path)
    (output / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="protocol changed"):
        complete_run(output, payload, {"arrays.npz": b"synthetic bytes"})
    assert not (output / "receipt.json").exists()


@pytest.mark.parametrize(
    "fault", ["receipt", "artifact", "protocol", "snapshot", "extra", "current_code"]
)
def test_receipt_tampering_and_current_source_drift_rejected(tmp_path, fault):
    output, source, payload = _reserved(tmp_path)
    receipt = complete_run(output, payload, {"arrays.npz": b"synthetic bytes"})
    if fault == "receipt":
        content = json.loads(receipt.read_text())
        content["synthetic"] = False
        receipt.write_text(json.dumps(content))
    elif fault == "artifact":
        (output / "arrays.npz").write_bytes(b"changed")
    elif fault == "protocol":
        (output / "protocol.json").write_text("{}")
    elif fault == "snapshot":
        (output / "source_script.py").write_text("changed")
    elif fault == "extra":
        (output / "unknown.txt").write_text("changed")
    elif fault == "current_code":
        source.write_text("changed")
        assert verify_run(output, check_current_sources=False)["synthetic"] is True
    with pytest.raises(ValueError):
        verify_run(output)


def test_unreserved_receipt_and_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="reservation"):
        complete_run(tmp_path, {}, {})
    output, _, payload = _reserved(tmp_path)
    with pytest.raises(ValueError, match="filenames"):
        complete_run(output, payload, {"../escape": b"bad"})
    assert not (tmp_path / "escape").exists()
