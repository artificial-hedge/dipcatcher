"""Diagnostics report and bundle: presence-only, redacted, locally packed."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import fx1
from fx1.harness import HARNESS_REGISTRY
from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS
from fx1.interactive.diagnostics import (
    DIAGNOSTIC_ENV_VARS,
    EXECUTABLE_NAMES,
    TELEMETRY_TAIL,
    collect_diagnostics,
    format_report,
    harness_registry_digest,
    redact_jsonl,
    write_bundle,
)
from fx1.interactive.runctx import new_id
from fx1.interactive.telemetry import Telemetry

NOW = datetime(2026, 10, 12, 2, 45, tzinfo=UTC)
SID = new_id("s", now=NOW, rand="abc12345")
API_KEY = "sk-" + "AbCdEfGhIjKlMnOpQrSt"
SENTINEL = "zzPromptSentinelzz"
PROXY_PASS = "zzProxyPass991zz"
TOKEN = "zzSigstoreToken991zz"


def _expected_digest() -> str:
    rows = sorted(
        (
            command.name,
            str(command.role),
            tuple(str(part) for part in command.argv),
            int(command.timeout_s),
        )
        for command in HARNESS_REGISTRY
    )
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "config_dir": tmp_path,
        "environ": {},
        "clock": lambda: NOW,
        "which": lambda _name: None,
    }
    options.update(overrides)
    return collect_diagnostics(**options)


def _slot(diag: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for item in diag["permissions"]:
        if isinstance(item, Mapping) and item.get("slot") == name:
            return item
    raise AssertionError(name)


def _zip_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name).decode("utf-8") for name in archive.namelist()}


def test_collect_runtime_harness_and_honesty(tmp_path: Path) -> None:
    diag = _base(tmp_path, session_id=SID, model="fx1")
    assert diag["generated_at"] == "2026-10-12T02:45:00.000000Z"
    assert diag["fx1_version"] == fx1.__version__
    assert diag["python"]["version"] == platform.python_version()
    assert diag["python"]["implementation"] == platform.python_implementation()
    assert diag["python"]["executable"] == sys.executable
    assert diag["platform"]["system"] == platform.system()
    assert set(diag["executables"]) == set(EXECUTABLE_NAMES)
    assert all(value is None for value in diag["executables"].values())
    assert diag["session_id"] == SID
    assert diag["session_id_rejected"] is False
    assert diag["model"] == "fx1"
    assert diag["harness"]["command_count"] == len(HARNESS_REGISTRY)
    assert diag["harness"]["registry_sha256"] == harness_registry_digest()
    assert diag["honesty"]["forbidden_headline_tokens"] == sorted(FORBIDDEN_HEADLINE_TOKENS)
    assert diag["endpoints"] == []
    assert diag["jobs"] is None
    assert diag["sessions"] is None
    assert diag["terminal"] is None
    assert diag["telemetry"]["events"] == []
    assert diag["environment"] == []


def test_registry_digest_is_stable() -> None:
    first = harness_registry_digest()
    second = harness_registry_digest()
    assert first == second == _expected_digest()
    assert len(first) == 64


def test_missing_callables_and_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_home(*_args: Any, **_kwargs: Any) -> Path:
        raise AssertionError("home")

    monkeypatch.setattr(Path, "home", _no_home)
    diag = _base(tmp_path)
    assert diag["config_dir"] == str(tmp_path)
    assert _slot(diag, "credentials")["exists"] is False
    assert _slot(diag, "sessions")["exists"] is False
    assert _slot(diag, "jobs")["exists"] is False
    assert _slot(diag, "logs")["exists"] is False
    assert _slot(diag, "history")["exists"] is False
    report = format_report(diag)
    assert "dipcatcher diagnostics" in report
    assert "\x1b" not in report
    assert fx1.__version__ in report


def test_clock_converts_to_utc(tmp_path: Path) -> None:
    shifted = datetime(2026, 10, 12, 4, 45, tzinfo=timezone(timedelta(hours=2)))
    aware = _base(tmp_path, clock=lambda: shifted)
    assert aware["generated_at"] == "2026-10-12T02:45:00.000000Z"
    naive = _base(tmp_path, clock=lambda: datetime(2026, 10, 12, 2, 45, 1))
    assert naive["generated_at"] == "2026-10-12T02:45:01.000000Z"


def test_which_injection(tmp_path: Path) -> None:
    diag = _base(tmp_path, which=lambda name: f"/opt/bin/{name}")
    assert diag["executables"]["dipcatcher"] == "/opt/bin/dipcatcher"
    assert diag["executables"]["quant"] == "/opt/bin/quant"
    assert diag["executables"]["fxi"] == "/opt/bin/fxi"
    assert diag["executables"]["fx1"] == "/opt/bin/fx1"


def test_permission_findings_do_not_read_file_bodies(tmp_path: Path) -> None:
    cred = tmp_path / "credentials.json"
    cred.write_text(json.dumps({"api_key": SENTINEL, "token": API_KEY}), encoding="utf-8")
    cred.chmod(0o644)
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.chmod(0o700)
    diag = _base(tmp_path)
    creds = _slot(diag, "credentials")
    problem = creds["problem"]
    assert creds["exists"] is True
    assert isinstance(problem, str)
    assert SENTINEL not in problem
    assert API_KEY not in problem
    assert _slot(diag, "logs")["exists"] is True
    assert _slot(diag, "logs")["problem"] is None
    blob = json.dumps(diag)
    assert SENTINEL not in blob
    assert API_KEY not in blob


def test_invalid_session_is_not_echoed(tmp_path: Path) -> None:
    bad = f"../{SENTINEL}"
    diag = _base(tmp_path, session_id=bad, model=API_KEY)
    assert diag["session_id"] is None
    assert diag["session_id_rejected"] is True
    blob = json.dumps(diag)
    report = format_report(diag)
    assert SENTINEL not in blob
    assert SENTINEL not in report
    assert API_KEY not in blob
    assert API_KEY not in report
    assert "session_id_rejected: true" in report


def test_env_names_only_and_proxy_reduced(tmp_path: Path) -> None:
    env = {
        "MOONSHOT_API_KEY": API_KEY,
        "FX1_BASE_URL": f"https://alicezz:{SENTINEL}@api.example/v1",
        "DIPCATCHER_OBSERVE": "1",
        "DIPCATCHER_SIGSTORE_ID_TOKEN": TOKEN,
        "HTTPS_PROXY": f"https://alicezz:{PROXY_PASS}@proxy.example:8443/path",
        "HTTP_PROXY": API_KEY,
        "ALL_PROXY": "http://[::1]:8080",
        "NO_PROXY": "localhost,secret.internal",
        "UNRELATED_SECRET": SENTINEL,
        "PATH": "/tmp",
    }
    diag = _base(tmp_path, environ=env)
    names = [row["name"] for row in diag["environment"]]
    assert "MOONSHOT_API_KEY" in names
    assert "FX1_BASE_URL" in names
    assert "DIPCATCHER_OBSERVE" in names
    assert "DIPCATCHER_SIGSTORE_ID_TOKEN" in names
    assert "NO_PROXY" in names
    assert "UNRELATED_SECRET" not in names
    assert "PATH" not in names
    for row in diag["environment"]:
        assert set(row) <= {"name", "set", "endpoint"}
        assert row["set"] is True
    https = next(row for row in diag["environment"] if row["name"] == "HTTPS_PROXY")
    assert https["endpoint"] == "https://proxy.example:8443"
    assert "@" not in str(https["endpoint"])
    http = next(row for row in diag["environment"] if row["name"] == "HTTP_PROXY")
    assert "endpoint" not in http
    all_proxy = next(row for row in diag["environment"] if row["name"] == "ALL_PROXY")
    assert all_proxy["endpoint"] == "http://[::1]:8080"
    blob = json.dumps(diag)
    report = format_report(diag)
    for secret in (API_KEY, SENTINEL, PROXY_PASS, TOKEN, "alicezz", "secret.internal"):
        assert secret not in blob
        assert secret not in report
    assert "MOONSHOT_API_KEY" in DIAGNOSTIC_ENV_VARS
    assert "proxy.example" in blob


def test_endpoint_allowlist_and_counts(tmp_path: Path) -> None:
    def rows() -> list[dict[str, str]]:
        return [
            {
                "model": "fx1",
                "host": "api.example",
                "status": "ready",
                "fingerprint": "wxyz",
                "api_key": API_KEY,
                "base_url": f"https://alicezz:{SENTINEL}@api.example/v1",
            },
            {"model": API_KEY, "host": "other.example", "status": "down", "fingerprint": "abcd"},
            "skip-me",  # type: ignore[list-item]
        ]

    def counts() -> dict[str, Any]:
        return {"running": 2, "queued": 0, "flag": True, "secret": SENTINEL}

    diag = _base(
        tmp_path,
        endpoint_rows=rows,
        job_counts=counts,
        session_counts=lambda: {"active": 1},
        terminal_caps={"color": "yes", "title": f"term {API_KEY}"},
    )
    assert diag["endpoints"] == [
        {"fingerprint": "wxyz", "host": "api.example", "model": "fx1", "status": "ready"},
        {"fingerprint": "abcd", "host": "other.example", "model": "[REDACTED]", "status": "down"},
    ]
    assert diag["jobs"] == {"queued": 0, "running": 2}
    assert diag["sessions"] == {"active": 1}
    blob = json.dumps(diag)
    assert API_KEY not in blob
    assert SENTINEL not in blob
    assert "alicezz" not in blob
    assert "api.example" in blob
    report = format_report(diag)
    assert "running: 2" in report
    assert API_KEY not in report


def test_callable_errors_omit_messages(tmp_path: Path) -> None:
    def boom() -> dict[str, int]:
        raise RuntimeError(f"leak {SENTINEL} {API_KEY}")

    diag = _base(tmp_path, job_counts=boom, session_counts=boom, endpoint_rows=boom)
    blob = json.dumps(diag)
    report = format_report(diag)
    assert SENTINEL not in blob
    assert API_KEY not in blob
    assert SENTINEL not in report
    assert API_KEY not in report
    assert diag["jobs"] == {"error_type": "RuntimeError", "ok": False}
    assert diag["sessions"]["error_type"] == "RuntimeError"
    assert diag["endpoints"] == [{"error_type": "RuntimeError", "status": "unavailable"}]


def test_telemetry_tail_is_capped_and_redacted(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    tel = Telemetry(session_id=SID, directory=logs, clock=lambda: NOW)
    for i in range(TELEMETRY_TAIL + 20):
        tel.emit("app.tick", {"i": i, "prompt": f"{SENTINEL}-{i}"})
    diag = _base(tmp_path, telemetry=tel, session_id=SID)
    events = diag["telemetry"]["events"]
    assert len(events) == TELEMETRY_TAIL
    assert events[0]["i"] == 20
    assert events[-1]["i"] == TELEMETRY_TAIL + 19
    assert diag["telemetry"]["dropped"] == 0
    assert diag["telemetry"]["level"] == "info"
    blob = json.dumps(diag)
    assert SENTINEL not in blob
    assert "[omitted:" in blob


def test_format_report_wraps_without_ansi(tmp_path: Path) -> None:
    marker = "Q" * 180
    diag = _base(tmp_path, terminal_caps={"columns": marker})
    text = format_report(diag, width=40)
    assert "\x1b" not in text
    assert "dipcatcher diagnostics" in text
    assert all(len(line) <= 40 for line in text.splitlines())
    assert marker not in text
    assert text.count("Q") == 180
    narrow = format_report(diag, width=5)
    assert all(len(line) <= 20 for line in narrow.splitlines())
    assert narrow.count("Q") == 180


def test_bundle_manifest_hashes_mode_and_no_overwrite(tmp_path: Path) -> None:
    diag = _base(tmp_path, session_id=SID, model="fx1-lite")
    path = tmp_path / "out" / "bundle.zip"
    assert write_bundle(diag, path) == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(path.parent.glob(".dipcatcher-diag.*")) == []
    members = _zip_members(path)
    assert set(members) == {"diagnostics.json", "report.txt", "MANIFEST.json"}
    manifest = json.loads(members["MANIFEST.json"])
    listed = {item["name"] for item in manifest["files"]}
    assert listed == {"diagnostics.json", "report.txt"}
    with zipfile.ZipFile(path) as archive:
        for item in manifest["files"]:
            data = archive.read(item["name"])
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            assert len(data) == item["bytes"]
    assert "\x1b" not in members["report.txt"]
    assert fx1.__version__ in members["report.txt"]
    assert SID in members["report.txt"]
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_bundle(diag, path)
    assert path.read_bytes() == saved


def test_bundle_redacts_telemetry_members(tmp_path: Path) -> None:
    first = tmp_path / "one" / f"{SID}.jsonl"
    second = tmp_path / "two" / f"{SID}.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    url = f"https://alicezz:{SENTINEL}@files.example/v1"
    first.write_text(
        json.dumps({"event": "model.response", "prompt": SENTINEL, "api_key": API_KEY, "note": url})
        + "\n"
        + f"NOTJSON {SENTINEL} {API_KEY}\n",
        encoding="utf-8",
    )
    second.write_text(json.dumps({"stdout": SENTINEL, "api_key": TOKEN}) + "\n", encoding="utf-8")
    folder = tmp_path / "events.jsonl"
    folder.mkdir()
    path = tmp_path / "support.zip"
    write_bundle(
        {"prompt": SENTINEL, "api_key": API_KEY, "note": f"Bearer {API_KEY}"},
        path,
        telemetry_files=[first, second, folder],
    )
    members = _zip_members(path)
    text = "\n".join(members.values())
    raw = path.read_bytes()
    for secret in (API_KEY, SENTINEL, TOKEN, "alicezz", "NOTJSON"):
        assert secret not in text
        assert secret.encode() not in raw
    assert "telemetry.unparsed" in text
    assert "telemetry.unreadable" in text
    assert "files.example" in text
    telemetry_names = [name for name in members if name.startswith("telemetry/")]
    assert len(telemetry_names) == 3
    assert "MANIFEST.json" in members
    manifest = json.loads(members["MANIFEST.json"])
    with zipfile.ZipFile(path) as archive:
        for item in manifest["files"]:
            data = archive.read(item["name"])
            assert hashlib.sha256(data).hexdigest() == item["sha256"]


def test_redact_jsonl_helpers() -> None:
    assert redact_jsonl("\n\n") == ""
    rendered = redact_jsonl(f"not-json {SENTINEL}\n" + json.dumps({"prompt": SENTINEL}))
    assert SENTINEL not in rendered
    assert "telemetry.unparsed" in rendered
    assert "[omitted:" in rendered


def test_secrets_absent_from_collect_report_and_zip(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    tel = Telemetry(session_id=SID, directory=logs, clock=lambda: NOW)
    tel.emit(
        "model.response",
        {
            "prompt": f"hello {SENTINEL}",
            "api_key": API_KEY,
            "stderr": TOKEN,
            "note": f"Authorization: Bearer {API_KEY}",
            "url": f"https://alicezz:{PROXY_PASS}@files.example/v1",
        },
    )
    env = {
        "MOONSHOT_API_KEY": API_KEY,
        "DIPCATCHER_SIGSTORE_ID_TOKEN": TOKEN,
        "HTTPS_PROXY": f"https://alicezz:{PROXY_PASS}@proxy.example:8443",
        "FX1_BYOK_API_KEY": SENTINEL,
    }

    def endpoints() -> list[dict[str, str]]:
        return [
            {
                "model": "fx1",
                "host": "api.example",
                "status": "ready",
                "fingerprint": "wxyz",
                "api_key": API_KEY,
                "password": PROXY_PASS,
            }
        ]

    diag = collect_diagnostics(
        config_dir=tmp_path,
        session_id=SID,
        model="fx1",
        endpoint_rows=endpoints,
        job_counts=lambda: {"running": 1},
        session_counts=lambda: {"active": 1},
        telemetry=tel,
        environ=env,
        terminal_caps={"title": f"term {API_KEY}"},
        clock=lambda: NOW,
        which=lambda name: f"/opt/{API_KEY}/{name}" if name == "fx1" else None,
    )
    report = format_report(diag)
    bundle = tmp_path / "diag.zip"
    write_bundle(diag, bundle, telemetry_files=tel.log_files())
    pieces = [
        json.dumps(diag),
        report,
        *_zip_members(bundle).values(),
        bundle.read_bytes().decode("utf-8", "replace"),
    ]
    for secret in (API_KEY, SENTINEL, PROXY_PASS, TOKEN, "alicezz"):
        for piece in pieces:
            assert secret not in piece
    assert diag["endpoints"][0]["host"] == "api.example"
    assert diag["endpoints"][0]["fingerprint"] == "wxyz"
    assert "proxy.example" in json.dumps(diag)
    assert "[omitted:" in json.dumps(diag["telemetry"])


def test_source_does_not_import_terminal() -> None:
    source = Path(__file__).resolve().parents[2] / "src/fx1/interactive/diagnostics.py"
    text = source.read_text(encoding="utf-8")
    assert "import fx1.interactive.terminal" not in text
    assert "from fx1.interactive.terminal" not in text


def test_non_posix_bundle_skips_chmod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows skips chmod. Patch this module's ``os`` only so pathlib keeps POSIX paths."""
    calls: list[int] = []
    real = os.chmod

    def spy(path: Path | str, mode: int) -> None:
        calls.append(mode)
        real(path, mode)

    class _OsProxy:
        name = "nt"

        def __getattr__(self, item: str) -> Any:
            return getattr(os, item)

    monkeypatch.setattr(os, "chmod", spy)
    monkeypatch.setattr("fx1.interactive.diagnostics.os", _OsProxy())
    path = tmp_path / "bundle.zip"
    write_bundle(_base(tmp_path), path)
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert calls == []
