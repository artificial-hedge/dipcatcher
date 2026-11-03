"""Production write-owner fail-closed, research read-only mount, and prod-smoke."""

from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from quant_fund.api.app import _MAX_REQUEST_BYTES
from quant_fund.paper.ledger import PaperLedger, _atomic_write_text
from quant_fund.utils.operator_kill import OperatorKillError, write_operator_kill
from quant_fund.utils.write_owner import (
    WriteOwnerError,
    claim_write_lock,
    release_write_lock,
    require_write_owner,
)

_ROOT = Path(__file__).resolve().parents[3]
_UNSET = "must be set when DIPCATCHER_PROFILE is production"
_RATE = "DIPCATCHER_RATE_LIMIT_RPS must be set above 0 in production"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entrypoint():
    return _load(_ROOT / "docker" / "entrypoint.py", "dipcatcher_entrypoint_ops8")


def _smoke():
    return _load(_ROOT / "scripts" / "prod_smoke.py", "dipcatcher_prod_smoke_ops8")


@pytest.fixture(autouse=True)
def _clear_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    monkeypatch.delenv("DIPCATCHER_WRITE_OWNER", raising=False)
    monkeypatch.delenv("DIPCATCHER_DATA_ROOT", raising=False)
    release_write_lock()
    yield
    release_write_lock()


@pytest.mark.parametrize("profile", ["production", "prod"])
def test_production_unset_refuses_ledger_and_kill_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", profile)
    path = tmp_path / "kill_switch"
    with pytest.raises(OperatorKillError, match=_UNSET) as raised:
        write_operator_kill(path, "HALT_NEW_ORDERS")
    assert "DIPCATCHER_WRITE_OWNER" in str(raised.value)
    assert not path.exists()
    assert not (tmp_path / ".kill_switch.tmp").exists()
    with pytest.raises(WriteOwnerError, match=_UNSET):
        _atomic_write_text(tmp_path / "broker_state.json", "{}\n")
    with pytest.raises(WriteOwnerError, match=_UNSET):
        PaperLedger(tmp_path, "run-1")
    with pytest.raises(WriteOwnerError, match=_UNSET):
        claim_write_lock(tmp_path)
    assert not (tmp_path / ".dipcatcher-write.lock").exists()
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(tmp_path))
    from quant_fund.api.serve import _claim_write_owner

    with pytest.raises(SystemExit, match="DIPCATCHER_WRITE_OWNER"):
        _claim_write_owner()


def test_blank_owner_is_unset_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", " Production ")
    monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "  ")
    with pytest.raises(WriteOwnerError, match=_UNSET):
        require_write_owner()


@pytest.mark.parametrize("profile", ["", "lab", "production-lab"])
def test_non_production_unset_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    if profile:
        monkeypatch.setenv("DIPCATCHER_PROFILE", profile)
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "ENABLED")
    assert path.read_text(encoding="utf-8").strip() == "ENABLED"
    _atomic_write_text(tmp_path / "broker_state.json", "{}\n")
    claim_write_lock(tmp_path)
    assert not (tmp_path / ".dipcatcher-write.lock").exists()


def test_production_reader_keeps_the_reader_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "0")
    with pytest.raises(WriteOwnerError, match="is not the writer") as raised:
        require_write_owner()
    assert "must be set" not in str(raised.value)
    claim_write_lock(tmp_path)
    assert not (tmp_path / ".dipcatcher-write.lock").exists()


def test_production_owner_writes_after_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", "prod")
    monkeypatch.setenv("DIPCATCHER_WRITE_OWNER", "1")
    with pytest.raises(WriteOwnerError, match="has not claimed the lock"):
        require_write_owner()
    claim_write_lock(tmp_path)
    _atomic_write_text(tmp_path / "broker_state.json", "{}\n")
    assert (tmp_path / "broker_state.json").read_text(encoding="utf-8") == "{}\n"


def test_production_research_refuses_a_writable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "50")
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "0")
    called: list[str] = []

    def _exec(cmd: str, args: list[str]) -> None:
        called.append(cmd)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    assert entry.main(["entrypoint", "python", "-m", "quant_fund.api.research_api"]) == 1
    assert called == []
    message = capsys.readouterr().err.strip()
    assert f"research data directory {data} must be read-only" in message
    assert "DIPCATCHER_RATE_LIMIT_RPS" not in message


def test_production_research_allows_a_read_only_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "prod")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "10")
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "1")
    seen: list[list[str]] = []

    def _exec(cmd: str, args: list[str]) -> None:
        seen.append(args)
        raise SystemExit(0)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    try:
        with pytest.raises(SystemExit) as raised:
            entry.main(["entrypoint", "python", "-m", "quant_fund.api.research_api"])
    finally:
        data.chmod(0o755)
    assert raised.value.code == 0
    assert seen == [["python", "-m", "quant_fund.api.research_api"]]


def test_production_research_reports_rate_and_writable_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "0")

    def _exec(cmd: str, args: list[str]) -> None:
        raise AssertionError(cmd)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    assert entry.main(["entrypoint", "python", "-m", "quant_fund.api.research_api"]) == 1
    message = capsys.readouterr().err
    assert _RATE in message
    assert "must be read-only" in message


def test_research_probe_errno_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    path = str(data)
    monkeypatch.setattr(entry, "_probe_errno", lambda _path: errno.EROFS)
    assert entry.check_research_data_dir(path) is None
    monkeypatch.setattr(entry, "_probe_errno", lambda _path: errno.EACCES)
    assert entry.check_research_data_dir(path) is None
    monkeypatch.setattr(entry, "_probe_errno", lambda _path: None)
    refused = entry.check_research_data_dir(path)
    assert refused is not None and "must be read-only" in refused
    monkeypatch.setattr(entry, "_probe_errno", lambda _path: errno.ENOSPC)
    assert entry.check_research_data_dir(path) == (
        f"data directory {path} is full; the dipcatcher user is uid 10001"
    )


def test_compose_research_mount_is_read_only_and_neither_replica_writes() -> None:
    text = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert text.count("/app/data:ro") == 1
    assert loaded["services"]["lab-a"]["volumes"][0].endswith(":/app/data")
    assert loaded["services"]["research-a"]["volumes"][0].endswith(":/app/data:ro")
    assert loaded["services"]["research-b"]["volumes"][0].endswith(":/app/data:ro")
    assert loaded["services"]["lab-a"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "1"
    assert loaded["services"]["lab-b"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"
    assert loaded["services"]["research-a"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"
    assert loaded["services"]["research-b"]["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"


def test_prod_smoke_target_lives_in_deploy_mk() -> None:
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "prod-smoke:" in prod
    assert "scripts/prod_smoke.py" in prod
    assert "prod-smoke:" not in root
    runbook = (_ROOT / "docs" / "OPS_RUNBOOK_PROD.md").read_text(encoding="utf-8")
    for heading in ("## Restart", "## Rollback", "## Certificate renewal", "## Replica failure"):
        assert heading in runbook
    assert "docs/OPS_RUNBOOK_PROD.md" in (_ROOT / "docs" / "PRODUCTION_RUNBOOK.md").read_text(
        encoding="utf-8"
    )


def test_smoke_helpers_and_body_limit() -> None:
    script = _smoke()
    assert script._BODY_LIMIT == _MAX_REQUEST_BYTES == 64 * 1024
    assert script.health_ok(
        200, {"Strict-Transport-Security": "max-age=31536000; includeSubDomains"}
    )
    assert not script.health_ok(200, {})
    assert not script.health_ok(503, {"Strict-Transport-Security": "max-age=31536000"})
    assert script.body_ok(405, 413)
    assert not script.body_ok(413, 413)
    assert not script.body_ok(500, 413)
    assert not script.body_ok(405, 200)
    assert script.research_mount_ok(str(errno.EROFS))
    assert not script.research_mount_ok(str(errno.EACCES))
    assert not script.research_mount_ok("writable")
    report = script.summarize(0, 0, [{"ok": True}])
    assert report["ok"] is True
    assert script.summarize(0, 0, [])["ok"] is False
    assert script.summarize(0, 0, [{"ok": False}])["ok"] is False
    assert script.summarize(1, 0, [{"ok": True}])["ok"] is False
    blob = script._CLAIM + script._READER + script._RESEARCH_PROBE
    assert "QUANT_API_KEY" not in blob
    assert "RESEARCH_API_KEY" not in blob


def test_smoke_refuses_arguments_and_missing_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _smoke()
    called: list[str] = []
    monkeypatch.setattr(script, "_make", lambda *_args: called.append("make") or 0)
    assert script.main(["prod_smoke.py", "--help"]) == 1
    assert called == []
    assert json.loads(capsys.readouterr().out)["error"] == "prod-smoke takes no arguments"
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.setenv("RESEARCH_API_KEY", "metrics-test-key-0123456789")
    assert script.main(["prod_smoke.py"]) == 1
    assert called == []
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "error": "QUANT_API_KEY and RESEARCH_API_KEY are required",
        "ok": False,
    }


def test_smoke_failure_still_runs_prod_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _smoke()
    monkeypatch.setenv("QUANT_API_KEY", "planted-test-key-value-0123456789")
    monkeypatch.setenv("RESEARCH_API_KEY", "metrics-test-key-0123456789")
    calls: list[str] = []

    def _make(target: str, _timeout: float) -> int:
        calls.append(target)
        return 0 if target == "prod-down" else 7

    monkeypatch.setattr(script, "_make", _make)

    def _checks() -> list[dict[str, object]]:
        raise AssertionError("checks")

    monkeypatch.setattr(script, "collect_checks", _checks)
    assert script.main(["prod_smoke.py"]) == 1
    assert calls == ["prod-up", "prod-down"]
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["ok"] is False
    assert payload["up"] == 7
    assert payload["down"] == 0
    assert payload["checks"] == []
    assert "planted-test-key" not in captured


def test_smoke_passes_and_redacts_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _smoke()
    key = "planted-test-key-value-0123456789"
    other = "metrics-test-key-0123456789"
    monkeypatch.setenv("QUANT_API_KEY", key)
    monkeypatch.setenv("RESEARCH_API_KEY", other)

    class _Proc:
        returncode = 0
        stdout = f"built {key}\n"
        stderr = other

    monkeypatch.setattr(script.subprocess, "run", lambda *_args, **_kwargs: _Proc())
    monkeypatch.setattr(
        script,
        "collect_checks",
        lambda: [{"name": "lab-health", "ok": True, "status": 200}],
    )
    assert script.main(["prod_smoke.py"]) == 0
    captured = capsys.readouterr()
    assert key not in captured.out
    assert other not in captured.out
    assert key not in captured.err
    assert other not in captured.err
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["up"] == 0
    assert payload["down"] == 0
    assert payload["checks"][0]["status"] == 200


def test_smoke_body_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _smoke()
    monkeypatch.setenv("QUANT_API_KEY", "planted-test-key-value-0123456789")
    monkeypatch.setenv("RESEARCH_API_KEY", "metrics-test-key-0123456789")
    calls: list[str] = []

    def _make(target: str, _timeout: float) -> int:
        calls.append(target)
        return 0

    monkeypatch.setattr(script, "_make", _make)
    monkeypatch.setattr(
        script,
        "collect_checks",
        lambda: [{"name": "lab-body", "ok": False, "small": 405, "large": 200}],
    )
    assert script.main(["prod_smoke.py"]) == 1
    assert calls == ["prod-up", "prod-down"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["checks"][0]["large"] == 200
    blob = json.dumps(payload)
    assert "planted-test-key" not in blob
    assert "metrics-test-key" not in blob


def test_docker_argv_follows_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _smoke()
    monkeypatch.delenv("DIPCATCHER_DOCKER", raising=False)
    monkeypatch.delenv("SUDO", raising=False)
    assert script.docker_argv() == ["sudo", "docker"]
    monkeypatch.setenv("SUDO", "")
    assert script.docker_argv() == ["docker"]
    monkeypatch.setenv("DIPCATCHER_DOCKER", "sudo docker")
    assert script.docker_argv() == ["sudo", "docker"]
