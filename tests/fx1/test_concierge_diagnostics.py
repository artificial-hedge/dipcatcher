"""Startup checks, /diagnostics, /budget wording, and content-free telemetry."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from fx1.interactive.concierge import Concierge
from fx1.interactive.jobs import JobLedger, JobState
from fx1.interactive.runctx import new_session_id
from fx1.interactive.telemetry import Telemetry
from fx1.serve.backends import ToolCompletion
from tests.fx1.test_concierge import FakeBackend, make_concierge


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def test_startup_checks_are_empty_when_clean(env: Path) -> None:
    del env
    assert Concierge().startup_checks() == []


def test_startup_checks_report_store_health_and_an_interrupted_job(
    env: Path, tmp_path: Path
) -> None:
    del env
    from fx1.interactive import profiles

    path = profiles.set_endpoint("fx1", "stored-health-key", "https://models.example.test/v1")
    path.chmod(0o644)
    ledger = JobLedger()
    job = ledger.create(kind="superpower", goal="left unfinished")
    ledger.transition(job.job_id, JobState.RUNNING)
    corrupt = ledger.root / "not-a-job.json"
    corrupt.write_text("{", encoding="utf-8")
    lines = Concierge().startup_checks()
    text = _joined(lines)
    assert "stored-health-key" not in text
    assert "0600" in text or "warning:" in text
    assert job.job_id in text
    assert "not restarted" in text
    assert "corrupt job file" in text
    del tmp_path


def test_diagnostics_omits_planted_secrets_in_the_report_and_the_zip(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del env
    secret = "diag-secret-key-value"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)
    concierge, out = make_concierge(tmp_path)
    bundle = tmp_path / "bundle.zip"
    concierge.handle_line(f"/diagnostics --bundle {bundle}")
    text = _joined(out)
    assert secret not in text
    assert bundle.is_file()
    assert bundle.stat().st_mode & 0o077 == 0
    with zipfile.ZipFile(bundle) as archive:
        blob = "\n".join(
            archive.read(name).decode("utf-8", "replace") for name in archive.namelist()
        )
    assert secret not in blob
    out.clear()
    concierge.handle_line(f"/diagnostics --bundle {bundle}")
    assert "refusing to overwrite" in _joined(out)


def test_budget_show_states_tokens_are_not_a_spend_cap(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/budget")
    text = _joined(out)
    assert "unit price is unknown" in text
    assert "no spend cap" in text


def test_telemetry_events_are_content_free(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del env
    secret = "telemetry-secret-key"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)
    session_id = new_session_id()
    telemetry = Telemetry(session_id=session_id, directory=tmp_path / "logs")
    backend = FakeBackend(
        [ToolCompletion(content="pinball loss is 0.2", tool_calls=None, finish_reason="stop")]
    )
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    concierge, _out = make_concierge(tmp_path, backend=backend)
    concierge._telemetry = telemetry.emit  # noqa: SLF001
    concierge._telemetry_box = telemetry  # noqa: SLF001
    concierge.handle_line(f"/model {secret}")
    concierge.handle_line(f"please hide {secret}")
    blob = json.dumps(telemetry.tail(20))
    assert secret not in blob
    events = {item.get("event") for item in telemetry.tail(20)}
    assert "concierge.turn" in events
    assert "concierge.command" in events
