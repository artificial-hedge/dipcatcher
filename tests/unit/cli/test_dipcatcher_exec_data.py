"""``dipcatcher exec --data`` validation, repeatable paths, and stale sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.interactive.data_ux import Attachment, attachment_payload, file_sha256
from fx1.interactive.headless import ExecRequest, ExecUsageError, run_exec
from fx1.interactive.sessions import SessionStore

_SCHEMA = {
    "schema",
    "status",
    "model",
    "served_model",
    "answer",
    "usage",
    "latency_s",
    "evidence",
    "session_id",
    "warnings",
    "errors",
    "duration_s",
}


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
        "timestamp,realized,forecast_var\n"
        "2020-01-01T00:00:00+00:00,1.0,1.1\n"
        "2020-01-01T01:00:00+00:00,1.2,1.1\n",
        encoding="utf-8",
    )
    return path


def _row(path: Path) -> Attachment:
    resolved = path.resolve()
    return Attachment(
        path=str(resolved),
        kind="csv",
        label=resolved.stem,
        sha256=file_sha256(resolved),
        size=resolved.stat().st_size,
        rows=2,
        columns=("timestamp", "realized", "forecast_var"),
        stale=False,
        has_time=True,
        time_monotonic=True,
        has_nulls=False,
        has_available_time=False,
    )


def test_exec_symlink_stays_inside_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    outside = _csv(tmp_path / "outside.csv")
    escape = data / "escape.csv"
    escape.symlink_to(outside)
    with pytest.raises(ExecUsageError, match="stay inside data"):
        run_exec(
            ExecRequest(prompt="/help", datasets=(str(escape),), timeout_s=30),
            install_signals=False,
        )
    dangling = data / "gone.csv"
    dangling.symlink_to(data / "missing.csv")
    with pytest.raises(ExecUsageError, match="not found"):
        run_exec(
            ExecRequest(prompt="/help", datasets=("gone.csv",), timeout_s=30),
            install_signals=False,
        )
    real = _csv(data / "real.csv")
    alias = data / "alias.csv"
    alias.symlink_to(real)
    outcome = run_exec(
        ExecRequest(prompt="/data list", datasets=("alias.csv",), timeout_s=60),
        install_signals=False,
    )
    assert outcome.exit_code == 0
    assert "real.csv" in outcome.answer


def test_bad_dataset_exits_2(tmp_path: Path) -> None:
    from dipcatcher_cli.exec_cmd import main

    (tmp_path / "data").mkdir(exist_ok=True)
    missing = tmp_path / "data" / "missing.csv"
    with pytest.raises(ExecUsageError, match="not found"):
        run_exec(
            ExecRequest(prompt="/help", datasets=(str(missing),), timeout_s=30),
            install_signals=False,
        )
    with pytest.raises(SystemExit) as raised:
        main(["--data", str(tmp_path / "notes.txt"), "/help"])
    assert raised.value.code == 2


def test_repeatable_data_lists_both_files(tmp_path: Path) -> None:
    first = _csv(tmp_path / "data" / "alpha.csv")
    second = _csv(tmp_path / "data" / "beta.csv")
    outcome = run_exec(
        ExecRequest(
            prompt="/data list",
            datasets=(str(first), str(second)),
            timeout_s=60,
        ),
        install_signals=False,
    )
    assert outcome.exit_code == 0
    assert "alpha" in outcome.answer
    assert "beta" in outcome.answer
    payload = outcome.as_json()
    assert set(payload) == _SCHEMA
    assert outcome.warnings == ()


def test_stale_session_dataset_is_a_warning(tmp_path: Path) -> None:
    path = _csv(tmp_path / "data" / "panel.csv")
    store = SessionStore(tmp_path / "sessions")
    record = store.create(model="fx1", messages=[])
    record["attachments"] = attachment_payload([_row(path)])
    store.save(record)
    path.write_text(path.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
    outcome = run_exec(
        ExecRequest(prompt="/data list", session=str(record["id"]), timeout_s=30),
        session_store=store,
        install_signals=False,
    )
    blob = json.dumps(outcome.as_json())
    assert "dataset changed since attach" in blob
    assert any("dataset changed since attach" in note for note in outcome.warnings)
    assert outcome.evidence is not None
    assert "dataset changed since attach" in str(outcome.evidence.get("summary"))
    assert set(outcome.as_json()) == _SCHEMA
