"""SYNTHETIC causal narrative/contrastive correctness, never market evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.models.narrative_market import (
    Arm,
    EchoLink,
    NarrativeConfig,
    NarrativeLearner,
    NarrativeWindow,
    PositionProxy,
    ReturnExample,
    TextRevision,
    VoiceObservation,
    _alignment_torch,
    contrastive_alignment,
    snapshot_window,
    voice_do_covariance,
)

NOW = datetime(2025, 1, 1, 12, tzinfo=UTC)
SHA = hashlib.sha256(b"SYNTHETIC narrative generator").hexdigest()
HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="optional torch unavailable")


def window(index: int) -> NarrativeWindow:
    decision = NOW + timedelta(days=index)
    statement = TextRevision(
        f"say{index}",
        "r0",
        "SYNTHETIC_ASSET",
        "SYNTHETIC_INSTITUTION",
        "say",
        "institution reviews outlook sector",
        decision - timedelta(minutes=5),
        decision - timedelta(minutes=4),
        decision - timedelta(minutes=3),
        "SYNTHETIC_STATEMENT",
        SHA,
        True,
    )
    cue = "expansion accelerating" if index % 2 else "erosion slowing"
    echo = TextRevision(
        f"echo{index}",
        "r0",
        "SYNTHETIC_ASSET",
        "SYNTHETIC_NEWSWIRE",
        "echo",
        f"report confirms {cue}",
        decision - timedelta(minutes=2),
        decision - timedelta(minutes=1, seconds=30),
        decision - timedelta(minutes=1),
        "SYNTHETIC_ECHO",
        SHA,
        True,
    )
    link = EchoLink(
        f"link{index}",
        echo.document_id,
        statement.document_id,
        "SYNTHETIC_ASSET",
        "SYNTHETIC_INSTITUTION",
        decision - timedelta(minutes=1),
        decision - timedelta(seconds=30),
        "SYNTHETIC_EXPLICIT_LINK",
        SHA,
        True,
    )
    return NarrativeWindow(
        f"w{index}",
        f"episode{index}",
        "SYNTHETIC_ASSET",
        "SYNTHETIC_INSTITUTION",
        decision,
        (statement, echo),
        (link,),
    )


def example(index: int) -> ReturnExample:
    w = window(index)
    target = (0.02 if index % 2 else -0.02) + 0.001 * math.sin(index)
    return ReturnExample(
        w,
        target,
        w.decision_time + timedelta(hours=1),
        w.decision_time + timedelta(hours=1, minutes=1),
        "SYNTHETIC_OBSERVED_RETURN_LABEL",
        SHA,
        True,
    )


def config(**kwargs: Any) -> NarrativeConfig:
    return NarrativeConfig(
        horizon=timedelta(hours=1),
        embedding_dim=4,
        steps=25,
        embargo=timedelta(minutes=2),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    if not HAS_TORCH:
        yield
        return
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.fixture
def fitted() -> NarrativeLearner:
    return NarrativeLearner(config()).fit(
        tuple(example(i) for i in range(12)),
        fit_cutoff=NOW + timedelta(days=13),
        calibration=tuple(example(i) for i in range(14, 20)),
        calibration_cutoff=NOW + timedelta(days=20),
    )


def test_independent_return_aligned_objective_and_entropy_bound() -> None:
    z = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0], [0.0, -1.0]])
    outcomes = np.array([[-1.0, 0.0], [-0.7, 0.2], [1.0, 1.0], [0.5, 0.8]])
    tau, b = 0.4, 0.8
    ce, entropy = [], []
    for i in range(len(z)):
        neighbors = [j for j in range(len(z)) if j != i]
        unscaled = [
            math.exp(-sum((outcomes[i] - outcomes[j]) ** 2) / (2 * b * b)) for j in neighbors
        ]
        q = [value / sum(unscaled) for value in unscaled]
        likelihood = [math.exp(float(z[i] @ z[j]) / tau) for j in neighbors]
        p = [value / sum(likelihood) for value in likelihood]
        ce.append(-sum(a * math.log(c) for a, c in zip(q, p, strict=True)))
        entropy.append(-sum(a * math.log(a) for a in q))
    result = contrastive_alignment(z, outcomes, temperature=tau, bandwidth=b)
    assert result.mean_cross_entropy == pytest.approx(float(np.mean(ce)))
    assert result.mean_entropy_bound == pytest.approx(float(np.mean(entropy)))
    assert result.per_anchor_excess_kl == pytest.approx(
        tuple(a - c for a, c in zip(ce, entropy, strict=True))
    )
    assert result.mean_cross_entropy >= result.mean_entropy_bound
    for bad in (np.zeros((4, 2)), np.full((4, 2), math.inf), np.ones((2, 2))):
        with pytest.raises(ValueError):
            contrastive_alignment(bad, outcomes, temperature=tau, bandwidth=b)


@requires_torch
def test_torch_alignment_matches_independent_objective_and_has_gradient() -> None:
    import torch

    z = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0], [0.0, -1.0]])
    y = np.array([[-1.0], [-0.8], [1.0], [0.6]])
    c = config()
    latent = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    loss = _alignment_torch(latent, torch.tensor(y, dtype=torch.float64), c, torch)
    reference = contrastive_alignment(
        z, y, temperature=c.temperature, bandwidth=c.outcome_bandwidth
    )
    assert float(loss.detach()) == pytest.approx(reference.mean_cross_entropy)
    gradient = torch.autograd.grad(loss, latent)[0]
    assert torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0.01


def test_causal_text_link_delay_revisions_and_hash_invariance() -> None:
    w = window(0)
    original = snapshot_window(w, config())
    future_revision = replace(
        w.texts[0],
        revision_id="r1",
        text="future bankruptcy revised outlook",
        published_at=w.decision_time + timedelta(hours=1),
        ingested_at=w.decision_time + timedelta(hours=1, minutes=1),
    )
    changed = replace(w, texts=w.texts + (future_revision,))
    before = snapshot_window(changed, config())
    assert before.say == original.say and before.snapshot_sha256 == original.snapshot_sha256
    assert "future" not in before.say[0].text
    later = snapshot_window(
        replace(changed, decision_time=w.decision_time + timedelta(hours=2)), config()
    )
    assert later.say[0].revision_id == "r1" and "future" in later.say[0].text
    delayed = snapshot_window(w, config(publication_delay=timedelta(minutes=2)))
    assert delayed.say and not delayed.echo
    link_delayed = replace(
        w,
        links=(
            replace(
                w.links[0],
                published_at=w.decision_time + timedelta(seconds=1),
                ingested_at=w.decision_time + timedelta(seconds=2),
            ),
        ),
    )
    assert not snapshot_window(link_delayed, config()).echo
    ingest_delayed = replace(
        w,
        texts=(w.texts[0], replace(w.texts[1], ingested_at=w.decision_time + timedelta(minutes=1))),
    )
    assert not snapshot_window(ingest_delayed, config()).echo
    with pytest.raises(ValueError, match="ambiguous"):
        snapshot_window(
            replace(
                w,
                texts=w.texts
                + (replace(w.texts[0], revision_id="conflict", text="conflicting text"),),
            ),
            config(),
        )
    with pytest.raises(ValueError, match="institution"):
        replace(w, texts=(replace(w.texts[0], speaker_id="other institution"), w.texts[1]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rows": 5},
        {"steps": 0},
        {"embedding_dim": 1},
        {"temperature": 0.0},
        {"outcome_bandwidth": 0.001},
        {"publication_delay": timedelta(seconds=-1)},
        {"seed": True},
        {"pair_operation_budget": 0},
    ],
)
def test_resource_config_fail_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        NarrativeConfig(**kwargs)


def test_source_clock_label_and_schema_fail_closed_before_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = window(0)
    for kwargs in (
        {"synthetic": 1},
        {"text": ""},
        {"source_sha256": "unknown"},
        {"published_at": w.texts[0].event_time - timedelta(seconds=1)},
        {"ingested_at": w.texts[0].published_at - timedelta(seconds=1)},
        {"event_time": NOW.replace(tzinfo=None)},
    ):
        with pytest.raises(ValueError):
            replace(w.texts[0], **kwargs)
    with pytest.raises(ValueError):
        replace(w, texts=[w.texts[0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(w.links[0], statement_document_id=w.links[0].echo_document_id)
    rows = tuple(example(i) for i in range(6))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="unpublished"):
        NarrativeLearner(config()).fit(rows, fit_cutoff=NOW + timedelta(hours=1))
    with pytest.raises(ValueError, match="constructed"):
        NarrativeLearner(config()).fit(
            (replace(rows[0], constructed_from_text=True),) + rows[1:],
            fit_cutoff=NOW + timedelta(days=7),
        )
    with pytest.raises(ValueError, match="resource"):
        NarrativeLearner(config(pair_operation_budget=1)).fit(
            rows, fit_cutoff=NOW + timedelta(days=7)
        )
    with pytest.raises(ImportError, match="optional nn"):
        NarrativeLearner(config()).fit(rows, fit_cutoff=NOW + timedelta(days=7))


@requires_torch
def test_actual_encoder_learning_frozen_train_only_preprocessing_and_proper_scores(
    fitted: NarrativeLearner,
) -> None:
    meta = fitted.metadata()
    final = hashlib.sha256(
        json.dumps(
            [{"name": v.name, "shape": v.shape, "values": v.values} for v in fitted.parameters()],
            sort_keys=True,
            default=str,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert final != meta["initial_parameter_sha256"]["narrative"]
    assert (
        meta["loss_history"]["narrative"][-1]["return_aligned_ce"]
        < meta["loss_history"]["narrative"][0]["return_aligned_ce"]
    )
    assert (
        len(meta["loss_history"]["bow"])
        == len(meta["loss_history"]["no_echo"])
        == fitted.config.steps
    )
    assert all("future" not in token for token in fitted.vocabulary)
    assert (
        not meta["market_evidence"] and not meta["pretrained"] and not meta["official_reproduction"]
    )
    assert meta["data_rights"] == "UNVERIFIED" and meta["synthetic"]
    outcomes = fitted.evaluate(
        tuple(example(i) for i in range(21, 27)), asof=NOW + timedelta(days=28)
    )
    assert {v["arm"] for v in outcomes["outcomes"]} == {"narrative", "no_echo", "bow"}
    assert all(
        0 <= v["brier"] <= 1 and v["binary_log_score"] >= 0 and 0 <= v["ece"] <= 1
        for v in outcomes["outcomes"]
    )
    assert all(
        v["calibration_status"] == "PLATT_CALIBRATED_LATER_SPLIT" for v in outcomes["outcomes"]
    )
    assert (
        not outcomes["comparison"]["benefit_asserted"]
        and not outcomes["comparison"]["supervision_richness_matched"]
    )
    detached = fitted.metadata()
    detached["states"]["narrative"][0]["values"][0] = 1e6
    assert fitted.metadata()["states"]["narrative"][0]["values"][0] != 1e6
    with pytest.raises(FrozenInstanceError):
        fitted.parameters()[0].values = (0.0,)  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="frozen"):
        fitted.fit(tuple(example(i) for i in range(6)), fit_cutoff=NOW + timedelta(days=7))


@requires_torch
def test_heldout_direction_scores_match_independent_arithmetic(fitted: NarrativeLearner) -> None:
    holdout = tuple(example(i) for i in range(21, 29))
    asof = NOW + timedelta(days=30)
    result = fitted.evaluate(holdout, asof=asof)
    labels = [float(r.target_return > 0) for r in holdout]
    for outcome in result["outcomes"]:
        forecast = fitted.predict(tuple(r.window for r in holdout), asof=asof, arm=outcome["arm"])
        p = forecast.probability_up
        expected_brier = sum(
            (prob - label) ** 2 for prob, label in zip(p, labels, strict=True)
        ) / len(p)
        expected_log_score = -sum(
            label * math.log(min(max(prob, 1e-9), 1 - 1e-9))
            + (1 - label) * math.log1p(-min(max(prob, 1e-9), 1 - 1e-9))
            for prob, label in zip(p, labels, strict=True)
        ) / len(p)
        expected_ece = 0.0
        for index in range(10):
            members = [j for j, prob in enumerate(p) if min(int(prob * 10), 9) == index]
            if members:
                expected_ece += abs(sum(p[j] - labels[j] for j in members)) / len(p)
        assert outcome["brier"] == pytest.approx(expected_brier)
        assert outcome["binary_log_score"] == pytest.approx(expected_log_score)
        assert outcome["ece"] == pytest.approx(expected_ece)
    assert result["ece_is_proper_score"] is False


@requires_torch
def test_calibration_and_holdout_labels_cannot_update_encoder_or_training_scale(
    fitted: NarrativeLearner,
) -> None:
    alternate = NarrativeLearner(config()).fit(
        tuple(example(i) for i in range(12)),
        fit_cutoff=NOW + timedelta(days=13),
        calibration=tuple(
            replace(example(i), target_return=-example(i).target_return) for i in range(14, 20)
        ),
        calibration_cutoff=NOW + timedelta(days=20),
    )
    arms: tuple[Arm, ...] = ("narrative", "no_echo", "bow")
    for arm in arms:
        assert alternate.parameters(arm) == fitted.parameters(arm)
    assert alternate.metadata()["target_scale"] == fitted.metadata()["target_scale"]
    assert alternate.metadata()["target_mean"] == fitted.metadata()["target_mean"]
    assert alternate.metadata()["calibration"] != fitted.metadata()["calibration"]
    unchanged = fitted.model_sha256
    fitted.evaluate(
        tuple(replace(example(i), target_return=0.3) for i in range(21, 27)),
        asof=NOW + timedelta(days=28),
    )
    assert fitted.model_sha256 == unchanged


@requires_torch
def test_echo_influences_joint_encoder_but_no_echo_baseline_ignores_it(
    fitted: NarrativeLearner,
) -> None:
    w = window(21)
    flipped = replace(
        w, texts=(w.texts[0], replace(w.texts[1], text="report confirms erosion slowing"))
    )
    first = fitted.predict((w,), asof=NOW + timedelta(days=22))
    second = fitted.predict((flipped,), asof=NOW + timedelta(days=22))
    assert abs(first.probability_up[0] - second.probability_up[0]) > 0.01
    assert first.embeddings != second.embeddings
    noecho = fitted.predict((w,), asof=NOW + timedelta(days=22), arm="no_echo")
    other = fitted.predict((flipped,), asof=NOW + timedelta(days=22), arm="no_echo")
    assert noecho.probability_up == other.probability_up and noecho.embeddings == other.embeddings


@requires_torch
def test_future_suffix_does_not_change_fit_vocab_weights_or_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(example(i) for i in range(6))
    future = replace(
        rows[0].window.texts[0],
        revision_id="future_rev",
        text="neverknownnewtoken fraud rumor",
        published_at=NOW + timedelta(hours=2),
        ingested_at=NOW + timedelta(hours=3),
    )
    amended = (
        replace(rows[0], window=replace(rows[0].window, texts=rows[0].window.texts + (future,))),
    ) + rows[1:]
    first = NarrativeLearner(config()).fit(rows, fit_cutoff=NOW + timedelta(days=7))
    second = NarrativeLearner(config()).fit(amended, fit_cutoff=NOW + timedelta(days=7))
    assert first.vocabulary == second.vocabulary and first.model_sha256 == second.model_sha256
    w = window(8)
    future2 = replace(
        w.texts[0],
        revision_id="future",
        text="holdoutfutureunknown",
        published_at=w.decision_time + timedelta(hours=1),
        ingested_at=w.decision_time + timedelta(hours=2),
    )
    asof = NOW + timedelta(days=9)
    original = first.predict((w,), asof=asof)
    changed = first.predict((replace(w, texts=w.texts + (future2,)), window(10)), asof=asof)
    assert original.forecast_sha256 == changed.forecast_sha256
    labels_changed = replace(example(8), target_return=0.5)
    before = first.predict((labels_changed.window,), asof=asof)
    assert before.probability_up == original.probability_up


@requires_torch
def test_global_rng_threads_and_cpu_default_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()

    def forbidden_seed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("global seed must not change")

    monkeypatch.setattr(torch, "manual_seed", forbidden_seed)
    with torch.device("meta"):
        fitted = NarrativeLearner(replace(config(), steps=2)).fit(
            tuple(example(i) for i in range(6)), fit_cutoff=NOW + timedelta(days=7)
        )
        assert fitted.metadata()["device"] == "cpu"
    assert torch.equal(state, torch.get_rng_state()) and torch.get_num_threads() == threads


@requires_torch
def test_chronological_disjoint_calibration_holdout_and_overlapping_targets_fail_closed(
    fitted: NarrativeLearner,
) -> None:
    training = tuple(example(i) for i in range(6))
    with pytest.raises(ValueError, match="overlap"):
        NarrativeLearner(config()).fit(
            training,
            fit_cutoff=NOW + timedelta(days=7),
            calibration=(replace(example(8), window=replace(window(8), episode_id="episode0")),),
            calibration_cutoff=NOW + timedelta(days=9),
        )
    with pytest.raises(ValueError, match="document"):
        new = replace(window(8), texts=(replace(window(8).texts[0], document_id="say0"),), links=())
        NarrativeLearner(config()).fit(
            training,
            fit_cutoff=NOW + timedelta(days=7),
            calibration=(replace(example(8), window=new),),
            calibration_cutoff=NOW + timedelta(days=9),
        )
    with pytest.raises(ValueError, match="follow"):
        NarrativeLearner(config()).fit(
            training,
            fit_cutoff=NOW + timedelta(days=7),
            calibration=(example(6),),
            calibration_cutoff=NOW + timedelta(days=8),
        )
    with pytest.raises(ValueError, match="horizon"):
        NarrativeLearner(config()).fit(
            (
                replace(
                    training[0],
                    target_event_time=training[0].target_event_time + timedelta(minutes=1),
                    target_available_time=training[0].target_available_time + timedelta(minutes=1),
                ),
            )
            + training[1:],
            fit_cutoff=NOW + timedelta(days=7),
        )
    near = replace(
        example(0),
        window=replace(
            window(0),
            window_id="near",
            episode_id="near",
            decision_time=NOW + timedelta(minutes=30),
        ),
        target_event_time=NOW + timedelta(hours=1, minutes=30),
        target_available_time=NOW + timedelta(hours=1, minutes=31),
    )
    with pytest.raises(ValueError, match="intervals overlap"):
        NarrativeLearner(config()).fit(training + (near,), fit_cutoff=NOW + timedelta(days=7))
    with pytest.raises(ValueError, match="cutoff"):
        fitted.predict((window(19),), asof=NOW + timedelta(days=30))
    with pytest.raises(ValueError, match="unpublished"):
        fitted.evaluate((example(21),), asof=NOW + timedelta(days=21))


@requires_torch
def test_infeasible_calibration_is_explicit_not_fabricated() -> None:
    training = tuple(example(i) for i in range(6))
    cal = tuple(replace(example(i), target_return=0.02) for i in range(8, 12))
    fitted = NarrativeLearner(config()).fit(
        training,
        fit_cutoff=NOW + timedelta(days=7),
        calibration=cal,
        calibration_cutoff=NOW + timedelta(days=12),
    )
    forecast = fitted.predict((window(13),), asof=NOW + timedelta(days=14))
    assert forecast.calibration_status == "UNAVAILABLE_TOO_FEW_OR_SINGLE_CLASS"
    assert fitted.metadata()["calibration"]["narrative"]["slope"] == 1.0


@requires_torch
def test_persistence_full_hash_write_once_roundtrip_and_tampering(
    fitted: NarrativeLearner, tmp_path: Path
) -> None:
    path = tmp_path / "narrative.json"
    fitted.save(path)
    restored = NarrativeLearner.load(path)
    forecast = fitted.predict((window(21),), asof=NOW + timedelta(days=22))
    assert restored.model_sha256 == fitted.model_sha256
    assert (
        restored.predict((window(21),), asof=NOW + timedelta(days=22)).forecast_sha256
        == forecast.forecast_sha256
    )
    with pytest.raises(FileExistsError):
        fitted.save(path)
    payload = json.loads(path.read_text())
    payload["body"]["states"]["narrative"][0]["values"][0] += 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        NarrativeLearner.load(path)
    payload["body"]["market_evidence"] = True
    payload["model_sha256"] = hashlib.sha256(
        json.dumps(payload["body"], sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="honesty"):
        NarrativeLearner.load(path)


@requires_torch
@pytest.mark.parametrize(
    "case",
    [
        "synthetic_denial",
        "nonboolean_provenance",
        "late_training_target",
        "early_calibration",
        "late_document",
        "overlapping_episodes",
        "duplicate_tensor",
        "false_calibration_status",
        "wrong_target_scale",
        "incomplete_config",
        "missing_statement_parent",
    ],
)
def test_rehashed_invalid_model_cannot_bypass_causal_provenance_validation(
    fitted: NarrativeLearner, tmp_path: Path, case: str
) -> None:
    path = tmp_path / "malformed.json"
    fitted.save(path)
    payload = json.loads(path.read_text())
    body = payload["body"]
    if case == "synthetic_denial":
        body["synthetic"] = False
    elif case == "nonboolean_provenance":
        body["derived_annotations"] = "false"
    elif case == "late_training_target":
        body["training_rows"][0]["target_available_time"] = (NOW + timedelta(days=40)).isoformat()
    elif case == "early_calibration":
        body["fit_cutoff"] = (NOW + timedelta(days=15)).isoformat()
    elif case == "late_document":
        doc = body["training_rows"][0]["documents"][0]
        doc["published_at"] = (NOW + timedelta(hours=1)).isoformat()
        doc["ingested_at"] = (NOW + timedelta(hours=2)).isoformat()
    elif case == "overlapping_episodes":
        body["calibration_rows"][0]["episode_id"] = body["training_rows"][0]["episode_id"]
    elif case == "duplicate_tensor":
        body["states"]["narrative"].append(body["states"]["narrative"][0])
    elif case == "false_calibration_status":
        body["calibration"]["narrative"]["status"] = "UNAVAILABLE_NO_CALIBRATION"
    elif case == "wrong_target_scale":
        body["target_mean"] = 10.0
    elif case == "incomplete_config":
        del body["config"]["steps"]
    elif case == "missing_statement_parent":
        body["training_rows"][0]["links"][0]["statement_document_id"] = "invented_parent"
    for partition in ("training", "calibration"):
        body[f"{partition}_data_sha256"] = hashlib.sha256(
            json.dumps(
                body[f"{partition}_rows"], sort_keys=True, default=str, allow_nan=False
            ).encode()
        ).hexdigest()
    payload["model_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        NarrativeLearner.load(path)


@requires_torch
def test_long_tokens_and_later_available_statement_corrections_roundtrip(tmp_path: Path) -> None:
    training = []
    for index in range(6):
        row = example(index)
        statement = row.window.texts[0]
        corrected = replace(
            statement,
            revision_id="r1",
            text="x" * 512,
            published_at=row.window.decision_time - timedelta(minutes=1),
            ingested_at=row.window.decision_time - timedelta(seconds=30),
        )
        training.append(
            replace(row, window=replace(row.window, texts=row.window.texts + (corrected,)))
        )
    fitted = NarrativeLearner(replace(config(), steps=2)).fit(
        tuple(training), fit_cutoff=NOW + timedelta(days=7)
    )
    assert "x" * 512 in fitted.vocabulary
    path = tmp_path / "valid_corrections.json"
    fitted.save(path)
    restored = NarrativeLearner.load(path)
    w, asof = window(8), NOW + timedelta(days=9)
    assert fitted.predict((w,), asof=asof) == restored.predict((w,), asof=asof)


@requires_torch
def test_artifact_read_and_write_resource_limits_leave_no_partial_model(
    fitted: NarrativeLearner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.models.narrative_market as module

    monkeypatch.setattr(module, "_MODEL_BYTES", 10)
    path = tmp_path / "too_large.json"
    with pytest.raises(ValueError, match="resource"):
        fitted.save(path)
    assert not path.exists()
    path.write_bytes(b" " * 11)
    with pytest.raises(ValueError, match="resource"):
        NarrativeLearner.load(path)


def observed_proxies() -> tuple[
    tuple[NarrativeWindow, ...], tuple[VoiceObservation, ...], tuple[PositionProxy, ...]
]:
    windows = tuple(window(i) for i in range(3))
    observations = []
    proxies = []
    for i, w in enumerate(windows):
        for doc, tone in zip(w.texts, (float(i - 1), float(1 - i)), strict=True):
            observations.append(
                VoiceObservation(
                    doc.document_id,
                    doc.revision_id,
                    w.entity_id,
                    w.institution_id,
                    doc.voice,
                    tone,
                    "SYNTHETIC_OBSERVED_TONE_PROXY",
                    w.decision_time - timedelta(seconds=20),
                    w.decision_time - timedelta(seconds=10),
                    "SYNTHETIC_TONE_OBSERVER",
                    SHA,
                    True,
                )
            )
        proxies.append(
            PositionProxy(
                f"proxy{i}",
                w.texts[0].document_id,
                w.entity_id,
                w.institution_id,
                w.decision_time - timedelta(days=10),
                w.decision_time,
                w.decision_time + timedelta(hours=12),
                w.decision_time + timedelta(hours=13),
                float(10 + 10 * i),
                "shares",
                "disclosed_holdings_change",
                "SYNTHETIC_POSITION_OBSERVER",
                SHA,
                True,
                derived=True,
            )
        )
    return windows, tuple(observations), tuple(proxies)


def test_observed_covariance_is_independent_proxy_diagnostic_not_trade_intent() -> None:
    windows, observations, proxies = observed_proxies()
    result = voice_do_covariance(
        windows, observations, proxies, asof=NOW + timedelta(days=4), config=config()
    )
    assert result.status == "AVAILABLE" and result.say_pairs == result.echo_pairs == 3
    assert result.say_do_covariance == pytest.approx(10.0)
    assert result.echo_do_covariance == pytest.approx(-10.0)
    assert result.intent == "UNKNOWN" and not result.actual_trades and not result.market_evidence
    assert result.synthetic and result.derived
    assert len(result.input_sha256) == len(result.diagnostic_sha256) == 64


def test_statement_and_position_proxy_identifiers_have_distinct_namespaces() -> None:
    windows, observations, proxies = observed_proxies()
    proxies = tuple(
        replace(p, proxy_id=windows[(i + 1) % 3].texts[0].document_id)
        for i, p in enumerate(proxies)
    )
    result = voice_do_covariance(
        windows, observations, proxies, asof=NOW + timedelta(days=4), config=config()
    )
    assert result.say_pairs == result.echo_pairs == 3
    assert result.say_do_covariance == pytest.approx(10.0)


def test_missing_delayed_or_text_derived_Do_is_unavailable_and_future_suffix_invariant() -> None:
    windows, observations, proxies = observed_proxies()
    missing = voice_do_covariance(
        windows, observations, (), asof=NOW + timedelta(days=4), config=config()
    )
    assert missing.status == "UNAVAILABLE" and missing.say_do_covariance is None
    delayed = tuple(
        replace(p, published_at=NOW + timedelta(days=10), ingested_at=NOW + timedelta(days=11))
        for p in proxies
    )
    later = voice_do_covariance(
        windows, observations, delayed, asof=NOW + timedelta(days=4), config=config()
    )
    assert (
        later.input_sha256 == missing.input_sha256
        and later.diagnostic_sha256 == missing.diagnostic_sha256
    )
    derived = voice_do_covariance(
        windows,
        observations,
        tuple(replace(p, constructed_from_text=True) for p in proxies),
        asof=NOW + timedelta(days=4),
        config=config(),
    )
    assert derived.status == "UNAVAILABLE" and derived.say_pairs == 0
    assert "text_constructed_position_proxy_excluded" in derived.reasons
    after = voice_do_covariance(
        windows, observations, proxies, asof=NOW + timedelta(days=4), config=config()
    )
    assert after.status == "AVAILABLE"


def test_covariance_link_tone_revision_units_and_independence_fail_closed() -> None:
    windows, observations, proxies = observed_proxies()
    asof = NOW + timedelta(days=4)
    with pytest.raises(ValueError, match="alignment"):
        voice_do_covariance(
            windows,
            observations,
            (replace(proxies[0], institution_id="other"),) + proxies[1:],
            asof=asof,
            config=config(),
        )
    with pytest.raises(ValueError, match="comparable"):
        voice_do_covariance(
            windows,
            observations,
            (replace(proxies[0], units="USD"),) + proxies[1:],
            asof=asof,
            config=config(),
        )
    with pytest.raises(ValueError, match="ambiguous"):
        voice_do_covariance(
            windows, observations + observations[:1], proxies, asof=asof, config=config()
        )
    late_tone = tuple(
        replace(
            v, published_at=NOW + timedelta(days=3), ingested_at=NOW + timedelta(days=3, hours=1)
        )
        for v in observations
    )
    unavailable = voice_do_covariance(windows, late_tone, proxies, asof=asof, config=config())
    assert unavailable.status == "UNAVAILABLE"
    unmatched = tuple(replace(v, revision_id="unobserved_revision") for v in observations)
    assert (
        voice_do_covariance(windows, unmatched, proxies, asof=asof, config=config()).status
        == "UNAVAILABLE"
    )


def test_echo_derived_and_synthetic_observations_propagate_to_covariance() -> None:
    windows, observations, proxies = observed_proxies()
    windows = tuple(
        replace(
            w,
            texts=tuple(replace(d, synthetic=False, derived=False) for d in w.texts),
            links=tuple(replace(link, synthetic=False, derived=False) for link in w.links),
        )
        for w in windows
    )
    observations = tuple(
        replace(v, synthetic=v.voice == "echo", derived=v.voice == "echo") for v in observations
    )
    proxies = tuple(replace(v, synthetic=False, derived=False) for v in proxies)
    result = voice_do_covariance(
        windows, observations, proxies, asof=NOW + timedelta(days=4), config=config()
    )
    assert result.synthetic and result.derived
