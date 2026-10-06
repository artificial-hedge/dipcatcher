"""Model-registry audit evidence and isolated resource lifecycle checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import models_audit as audit
from fx1.serve.backends import BackendNotConfiguredError
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.models_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 157
    assert all(value is True for value in measured.values()), measured


def test_tombstone_and_durability_probes_hold(measured: dict[str, bool]) -> None:
    assert measured["ft_delete_tombstones_get"] is True
    assert measured["ft_delete_tombstones_list"] is True
    assert measured["ft_delete_tombstones_resolve"] is True
    assert measured["ft_delete_drops_checkpoints"] is True
    assert measured["inflight_completes_across_delete"] is True
    assert measured["inflight_new_request_404s"] is True
    assert measured["dur_registry_survives"] is True
    assert measured["dur_tombstone_stays_dead"] is True
    assert measured["dur_base_never_tombstoned"] is True


def test_registry_defect_regressions_stay_pinned(measured: dict[str, bool]) -> None:
    """The four defects this battery found stay fixed: honest ft card
    ``created``, no silent default link on unknown models, the ``ft:``
    alias on the response ``model``, and empty-model validation."""
    assert measured["ft_card_created_is_registration"] is True
    assert measured["param_unknown_chat_404"] is True
    assert measured["param_unknown_never_resolved"] is True
    assert measured["param_empty_chat_422"] is True
    assert measured["ft_response_model_is_ft_name"] is True
    assert measured["sdk_ft_response_model"] is True
    assert measured["ckpt_dir_only_local_fx1"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "models_audit", lambda: results)
    receipt = audit.models_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "models_audit", lambda: measured)
    receipt = audit.models_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.models_audit_bench()


def test_client_resources_and_ambient_state_survive_probe_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ambient = tmp_path / "operator"
    ambient.mkdir()
    sentinel = ambient / "ft_jobs.jsonl"
    sentinel.write_text("operator data must remain unchanged\n")
    config = {
        "FX1_API_STATE_DIR": str(ambient),
        "FX1_API_RECEIPTS_DIR": str(ambient),
        "FX1_FT_DIR": str(ambient),
        "FX1_SDK_STATE_DIR": str(ambient),
        "FX1_API_JOB_MAX": "invalid ambient value",
        "MOONSHOT_API_KEY": "synthetic ambient sentinel",
    }
    for name, value in config.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="deliberate failure"), audit._audit_context():
        assert all(name not in os.environ for name in config)
        temporary = audit._temporary_directory()
        client, _ = audit._client(audit._spy_resolver())
        assert client.get("/v1/models").status_code == 200
        executor = client.app.state.jobs_executor
        assert executor.submit(lambda: 2).result(timeout=1) == 2
        raise RuntimeError("deliberate failure")
    assert client.is_closed
    assert not temporary.exists()
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)
    assert all(os.environ[name] == value for name, value in config.items())
    assert list(ambient.iterdir()) == [sentinel]
    assert sentinel.read_text() == "operator data must remain unchanged\n"


def test_served_model_attribution_is_not_publicly_forgeable() -> None:
    """Registry attribution is evidence, not a caller-controlled label."""
    with audit._audit_context():
        resolver = audit._spy_resolver()
        client, _ = audit._client(resolver)
        response = client.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "hi"}],
                "served_model": "ft:forged:trusted:model",
            },
        )
        schema = client.app.openapi()["components"]["schemas"]["CompleteRequest"]
        assert response.status_code == 422
        assert resolver.resolved == []
        assert "served_model" not in schema["properties"]


def test_ft_fallback_reports_the_backend_that_really_served() -> None:
    """A failed checkpoint may fall back, but cannot inherit its ft alias."""
    with audit._audit_context():
        resolver = audit._spy_resolver()
        client, _ = audit._client(resolver)
        _job, model = audit._ft_job(client, suffix="fallback")
        resolver.raise_for["local_fx1"] = BackendNotConfiguredError("checkpoint unavailable")
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "fx1": {"fallbacks": ["hosted_k3"]},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "hosted-stub"
        assert response.json()["system_fingerprint"] == "hosted_k3"


def test_sdk_ft_fallback_reports_the_backend_that_really_served() -> None:
    from fx1.sdk import Fx1Harness

    resolver = audit._spy_resolver()
    resolver.raise_for["local_fx1"] = BackendNotConfiguredError("checkpoint unavailable")
    harness = Fx1Harness(backend_resolver=resolver)
    result = harness.complete(
        [{"role": "user", "content": "hi"}],
        backend="local_fx1",
        checkpoint_dir="/srv/registered-checkpoint",
        fallbacks=["hosted_k3"],
        _served_model="ft:fx1:fallback:registered",
    )
    assert result.backend == "hosted_k3"
    assert result.model == "hosted-stub"


@pytest.mark.parametrize("checkpoint", ["", "bad\x00checkpoint"])
def test_malformed_checkpoint_paths_are_client_refusals(checkpoint: str) -> None:
    with audit._audit_context():
        resolver = audit._spy_resolver()
        client, _ = audit._client(resolver)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local_fx1",
                "messages": [{"role": "user", "content": "hi"}],
                "fx1": {"backend": "local_fx1", "checkpoint_dir": checkpoint},
            },
        )
        assert response.status_code == 422
        assert resolver.resolved == []


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {"model": "x" * 513, "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/responses", {"model": "x" * 513, "input": "hi"}),
        (
            "/v1/messages",
            {
                "model": "x" * 513,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
    ],
)
def test_model_ids_are_bounded_before_routing(path: str, body: dict[str, Any]) -> None:
    with audit._audit_context():
        resolver = audit._spy_resolver()
        client, _ = audit._client(resolver)
        response = client.post(path, json=body)
        assert response.status_code == 422
        assert resolver.resolved == []


def _ft_record(job_id: str) -> Any:
    from fx1.serve.finetune import FTJob

    return FTJob(
        id=job_id,
        model="fx1",
        created_at=1,
        status="succeeded",
        training_file="file-training",
    )


def test_registry_journal_failures_do_not_publish_or_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fx1.serve.finetune import FTJobStore
    from fx1.serve.journal import JobJournal

    journal = JobJournal(tmp_path / "ft_jobs.jsonl")
    store = FTJobStore(2, journal=journal)
    store.put(_ft_record("job-0"), None, "fp-0")
    store.register_model("ft:fx1:a:job0", job_id="job-0", checkpoint="ckpt-0", created=1)
    original_append = journal.append

    def fail_append(_payload: dict[str, Any]) -> None:
        raise OSError("synthetic journal failure")

    monkeypatch.setattr(journal, "append", fail_append)
    with pytest.raises(OSError, match="synthetic journal failure"):
        store.register_model("ft:fx1:b:job0", job_id="job-0", checkpoint="ckpt-1", created=2)
    assert store.get_model("ft:fx1:b:job0") is None
    with pytest.raises(OSError, match="synthetic journal failure"):
        store.unregister_model("ft:fx1:a:job0")
    assert store.get_model("ft:fx1:a:job0") is not None
    with pytest.raises(OSError, match="synthetic journal failure"):
        store.put(_ft_record("job-1"), None, "fp-1")
    assert store.get("job-0") is not None
    assert store.get("job-1") is None
    monkeypatch.setattr(journal, "append", original_append)


def test_evicted_finetune_worker_cannot_resurrect_after_restart(tmp_path: Path) -> None:
    from fx1.serve.finetune import FTJobStore
    from fx1.serve.journal import JobJournal

    path = tmp_path / "ft_jobs.jsonl"
    store = FTJobStore(1, journal=JobJournal(path))
    evicted = store.put(_ft_record("job-old"), None, "old")
    store.put(_ft_record("job-new"), None, "new")
    evicted.job.finished_at = 2
    store.mark(evicted)
    store.register_model("ft:fx1:late:jobold", job_id="job-old", checkpoint="late-ckpt", created=2)

    recovered = FTJobStore(1, journal=JobJournal(path))
    assert recovered.get("job-old") is None
    assert recovered.get_model("ft:fx1:late:jobold") is None
    assert recovered.get("job-new") is not None
