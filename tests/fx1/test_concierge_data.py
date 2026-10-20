"""Concierge ``/data``, ``/runs``, and ``/models check`` without the lab scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.jobs import JobLedger, JobState
from fx1.interactive.sessions import SessionStore
from tests.fx1.stub_model_server import StubModelServer

_KEY = "sk-test-fx1-secret-value-aaa111"


class PathHarness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(
        self, name: str, extra_args: list[str] | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del kwargs
        args = list(extra_args or [])
        self.calls.append((name, args))
        path = Path(args[args.index("--data") + 1])
        payload = {
            "schema": "dataset_profile.v1",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "n_rows": 3,
            "columns": [
                {"name": "timestamp", "dtype": "String", "nulls": 0},
                {"name": "realized", "dtype": "Float64", "nulls": 0},
            ],
            "time": {
                "column": "timestamp",
                "parsed": True,
                "strictly_increasing": True,
                "start": "2020-01-01T00:00:00+00:00",
                "end": "2020-01-01T01:00:00+00:00",
            },
            "suggested_roles": {"available_time": {"column": None}},
        }
        return SimpleNamespace(exit_code=0, stdout=json.dumps(payload), stderr="")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def _csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,realized,forecast_var\n2020-01-01T00:00:00+00:00,1.0,1.1\n",
        encoding="utf-8",
    )
    return path


def _concierge(store: SessionStore, harness: PathHarness, out: list[str]) -> Concierge:
    return Concierge(
        session_store=store,
        harness=harness,
        approvals=ApprovalGate(),
        output_fn=out.append,
        animated=False,
        input_fn=lambda _prompt: "n",
    )


def test_attach_list_remove_resume_and_new(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    harness = PathHarness()
    out: list[str] = []
    concierge = _concierge(store, harness, out)
    path = _csv(tmp_path / "data" / "panel.csv")
    assert concierge.handle_line(f"/data attach {path}") is False
    text = "\n".join(out)
    assert text.splitlines()[0] == f"attached dataset {path.resolve()}"
    assert "profile rows=3 columns=timestamp, realized" in text
    assert "point-in-time validity will NOT be verified" in text
    out.clear()
    concierge.handle_line("/data list")
    assert "panel" in "\n".join(out)
    session_id = str(concierge._session["id"])  # noqa: SLF001
    saved = store.load(session_id)
    assert saved["attachments"][0]["label"] == "file"
    assert saved["attachments"][0]["sha256"]
    assert saved["datasets"] == [{"path": str(path.resolve()), "kind": "csv", "label": "file"}]
    other_out: list[str] = []
    other = _concierge(store, harness, other_out)
    other.handle_line(f"/resume {session_id[:8]}")
    other.handle_line("/data list")
    assert "panel" in "\n".join(other_out)
    other.handle_line("/data remove 1")
    other.handle_line("/new")
    other_out.clear()
    other.handle_line("/data list")
    assert "\n".join(other_out) == "no datasets attached"
    fresh = store.load(str(other._session["id"]))  # noqa: SLF001
    assert fresh["attachments"] == []


def test_superpower_refuses_a_changed_digest(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    harness = PathHarness()
    out: list[str] = []
    concierge = _concierge(store, harness, out)
    path = _csv(tmp_path / "data" / "panel.csv")
    concierge.handle_line(f"/data attach {path}")
    path.write_text(path.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
    out.clear()
    before = len(harness.calls)
    assert concierge.handle_line("/superpower score my forecasts --sync") is False
    text = "\n".join(out)
    assert "dataset changed since attach; /data inspect 1 then re-attach" in text
    assert all(name != "score-forecasts" for name, _args in harness.calls[before:])
    concierge.handle_line("/data inspect 1")
    assert "file changed since attach" in "\n".join(out)


def test_runs_are_read_only_after_restart(tmp_path: Path) -> None:
    ledger = JobLedger()
    record = ledger.create(kind="superpower", goal="qlike check")
    ledger.transition(record.job_id, JobState.RUNNING)
    step = ledger.begin_step(
        record.job_id,
        name="score-forecasts",
        argv=["score-forecasts", "--task", "volatility"],
        idempotent=False,
    )
    receipt = tmp_path / "forecast_scoring.json"
    receipt.write_text("{}", encoding="utf-8")
    ledger.finish_step(
        record.job_id,
        step.step_id,
        state=JobState.SUCCEEDED,
        exit_code=0,
        receipt_paths=[str(receipt)],
    )
    ledger.transition(record.job_id, JobState.SUCCEEDED)
    revision = ledger.get(record.job_id).revision
    out: list[str] = []
    concierge = _concierge(SessionStore(tmp_path / "sessions"), PathHarness(), out)
    concierge.handle_line("/runs")
    text = "\n".join(out)
    assert record.job_id in text
    assert "superpower" in text
    assert "succeeded" in text
    assert "steps 1/1" in text
    out.clear()
    concierge.handle_line(f"/runs {record.job_id[:12]}")
    detail = "\n".join(out)
    assert "score-forecasts" in detail
    assert "receipt" in detail
    assert str(receipt) in detail
    assert ledger.get(record.job_id).revision == revision
    restarted: list[str] = []
    again = _concierge(SessionStore(tmp_path / "sessions-2"), PathHarness(), restarted)
    again.handle_line("/runs")
    assert record.job_id in "\n".join(restarted)


def test_models_check_offline_and_live(tmp_path: Path) -> None:
    import re

    out: list[str] = []
    concierge = _concierge(SessionStore(tmp_path / "sessions"), PathHarness(), out)
    concierge.handle_line("/models check")
    text = "\n".join(out)
    assert "nothing was contacted" in text
    assert "not_contacted" in text
    assert _KEY not in text
    out.clear()
    with StubModelServer({"fx1": _KEY, "fx1-lite": _KEY}) as server:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", _KEY, server.url)
        profiles.set_endpoint("fx1-lite", _KEY, server.url)
        concierge.handle_line("/models check --live fx1")
        checked = "\n".join(out)
        assert _KEY not in checked
        assert "status=ok" in checked
        assert "served=fx1" in checked
        out.clear()
        concierge.handle_line("/models")
        table = "\n".join(out)
    assert _KEY not in table
    assert re.search(r"\d+\.\d{2}s", table)


def test_help_lists_data_and_runs(tmp_path: Path) -> None:
    out: list[str] = []
    concierge = _concierge(SessionStore(tmp_path / "sessions"), PathHarness(), out)
    concierge.handle_line("/help")
    text = "\n".join(out)
    assert "/data" in text
    assert "/runs" in text
    assert "/models" in text
