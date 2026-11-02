"""Production rate default, preflight, and the ledger restore drill."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from quant_fund.api.rate_limit import BUCKET, PRODUCTION_DEFAULT_RPS, configured_rps, reset_bucket
from quant_fund.api.service_config import ServiceConfigError, ServiceSettings
from quant_fund.paper.state_backup import PaperBackupError, backup_paper_state, restore_paper_state

_ROOT = Path(__file__).resolve().parents[3]
_NAMES = (
    "DIPCATCHER_PROFILE",
    "DIPCATCHER_RATE_LIMIT_RPS",
    "DIPCATCHER_REQUIRE_DATA_WRITE",
    "DIPCATCHER_WRITE_OWNER",
    "DIPCATCHER_BIND_HOST",
    "DIPCATCHER_BIND_PORT",
    "PORT",
    "QUANT_API_KEY",
    "RESEARCH_API_KEY",
    "RESEARCH_API_HOST",
    "RESEARCH_API_PORT",
    "RESEARCH_API_DATA_ROOT",
    "DIPCATCHER_OBSERVE",
    "DIPCATCHER_METRICS_HOST",
    "DIPCATCHER_METRICS_PORT",
    "DIPCATCHER_METRICS_KEY_ENV",
    "DIPCATCHER_METRICS_LOOPBACK_SCRAPE",
    "DIPCATCHER_LOG_LEVEL",
    "DIPCATCHER_SHUTDOWN_GRACE_S",
    "DIPCATCHER_DATA_ROOT",
    "DIPCATCHER_KILL_SWITCH_PATH",
    "DIPCATCHER_LAB_DATA",
    "DIPCATCHER_RESEARCH_DATA",
    "DIPCATCHER_TLS_DIR",
    "LAB_TLS_PORT",
    "RESEARCH_TLS_PORT",
    "PROMETHEUS_PUBLISH_PORT",
    "GRAFANA_PUBLISH_PORT",
    "SUDO",
    "PROJECT",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entrypoint():
    return _load(_ROOT / "docker" / "entrypoint.py", "dipcatcher_entrypoint_ops9")


def _preflight():
    return _load(_ROOT / "scripts" / "prod_preflight.py", "dipcatcher_prod_preflight_ops9")


@pytest.fixture(autouse=True)
def _clear(monkeypatch: pytest.MonkeyPatch):
    for name in ("DIPCATCHER_PROFILE", "DIPCATCHER_RATE_LIMIT_RPS", "DIPCATCHER_WRITE_OWNER"):
        monkeypatch.delenv(name, raising=False)
    reset_bucket()
    yield
    reset_bucket()


def test_production_unset_rate_stays_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("RESEARCH_API_HOST", "127.0.0.1")
    monkeypatch.setenv("DIPCATCHER_BIND_HOST", "127.0.0.1")
    assert ServiceSettings.for_research().service == "research"
    assert ServiceSettings.for_lab("127.0.0.1", 8000).service == "lab"
    assert configured_rps() == PRODUCTION_DEFAULT_RPS == 50.0
    answers = [BUCKET.take()[0] for _ in range(51)]
    assert answers[:50] == [True] * 50
    assert answers[50] is False


@pytest.mark.parametrize("profile", ["", "lab", "production-lab"])
def test_non_production_unset_rate_stays_off(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    if profile:
        monkeypatch.setenv("DIPCATCHER_PROFILE", profile)
    assert configured_rps() == 0.0
    assert BUCKET.take() == (True, 0)


def test_blank_production_rate_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", " Prod ")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "   ")
    assert configured_rps() == PRODUCTION_DEFAULT_RPS


def test_production_zero_still_refuses_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_PROFILE", "prod")
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("RESEARCH_API_HOST", "127.0.0.1")
    with pytest.raises(ServiceConfigError, match="must be set above 0 in production"):
        ServiceSettings.for_research()


def test_entrypoint_fills_an_unset_production_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entrypoint()
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(data))
    monkeypatch.setenv("DIPCATCHER_PROFILE", "production")
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "0")
    seen: list[str | None] = []

    def _exec(cmd: str, args: list[str]) -> None:
        seen.append(os.environ.get("DIPCATCHER_RATE_LIMIT_RPS"))
        raise SystemExit(0)

    monkeypatch.setattr(entry.os, "execvp", _exec)
    try:
        with pytest.raises(SystemExit) as raised:
            entry.main(["entrypoint", "python", "-m", "quant_fund.api.research_api"])
    finally:
        data.chmod(0o755)
    assert raised.value.code == 0
    assert seen == [str(int(PRODUCTION_DEFAULT_RPS))]
    assert str(int(PRODUCTION_DEFAULT_RPS)) == entry._PRODUCTION_DEFAULT_RPS
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    monkeypatch.delenv("DIPCATCHER_RATE_LIMIT_RPS", raising=False)
    entry._apply_production_rate_default()
    assert os.environ.get("DIPCATCHER_RATE_LIMIT_RPS") in {None, ""}


def test_prod_env_example_lists_the_production_surface() -> None:
    example = (_ROOT / "deploy" / "prod.env.example").read_text(encoding="utf-8")
    root = (_ROOT / ".env.example").read_text(encoding="utf-8")
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    for name in _NAMES:
        assert name in example
    assert "default:" in example
    assert "API_KEY=" not in example
    assert "BEGIN PRIVATE KEY" not in example
    assert "DIPCATCHER_PROFILE" not in root
    assert "prod-up: prod-certs prod-data prod-preflight prod-build" in prod
    for name in ("prod-preflight", "prod-backup", "prod-restore"):
        assert f"{name}:" in prod
        assert f"{name}:" not in makefile
    runbook = (_ROOT / "docs" / "OPS_RUNBOOK_PROD.md").read_text(encoding="utf-8")
    assert "## Backup and restore" in runbook
    assert "## Preflight" in runbook


def test_clock_and_ports_and_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _preflight()
    now = 1_762_000_000.0
    assert script.clock_ok(now, now - 10, now + 40 * 86400)
    assert not script.clock_ok(0.0, -10, 40 * 86400)
    assert not script.clock_ok(now, now + 180, now + 40 * 86400)
    assert not script.clock_ok(now, now - 10, now + 1000)
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    busy = held.getsockname()[1]
    try:
        assert script.port_free(busy) is False
    finally:
        held.close()
    assert script.port_free(busy) is True
    monkeypatch.setattr(script.shutil, "disk_usage", lambda _path: type("U", (), {"free": 0})())
    assert script.check_disk([tmp_path])["ok"] is False


def _mint(directory: Path) -> None:
    directory.mkdir()
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "30",
            "-nodes",
            "-keyout",
            str(directory / "dev.key"),
            "-out",
            str(directory / "dev.crt"),
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    (directory / "dev.key").chmod(0o600)


def test_preflight_json_and_a_refused_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _preflight()
    tls = tmp_path / "tls"
    lab = tmp_path / "lab"
    research = tmp_path / "research"
    _mint(tls)
    lab.mkdir()
    research.mkdir()
    monkeypatch.setenv("DIPCATCHER_TLS_DIR", str(tls))
    monkeypatch.setenv("DIPCATCHER_LAB_DATA", str(lab))
    monkeypatch.setenv("DIPCATCHER_RESEARCH_DATA", str(research))
    monkeypatch.setenv("QUANT_API_KEY", "planted-test-key-value-0123456789")
    monkeypatch.setenv("RESEARCH_API_KEY", "metrics-test-key-0123456789")
    free = _free_ports(4)
    monkeypatch.setenv("LAB_TLS_PORT", str(free[0]))
    monkeypatch.setenv("RESEARCH_TLS_PORT", str(free[1]))
    monkeypatch.setenv("PROMETHEUS_PUBLISH_PORT", str(free[2]))
    monkeypatch.setenv("GRAFANA_PUBLISH_PORT", str(free[3]))
    assert script.main(["prod_preflight.py"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {item["name"] for item in payload["checks"]} == {
        "env",
        "certs",
        "ports",
        "disk",
        "write-owner",
        "clock",
    }
    blob = json.dumps(payload)
    assert "planted-test-key-value-0123456789" not in blob
    assert "metrics-test-key-0123456789" not in blob
    monkeypatch.setenv("QUANT_API_KEY", "planted test-key value 0123456789")
    assert script.main(["prod_preflight.py"]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["ok"] is False
    assert "planted test-key value 0123456789" not in json.dumps(failed)
    assert script.main(["prod_preflight.py", "--help"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def _free_ports(count: int) -> list[int]:
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def test_restore_drill_keeps_a_divergent_kill_file(tmp_path: Path) -> None:
    source = tmp_path / "lab"
    run = source / "metadata" / "paper" / "run-1"
    run.mkdir(parents=True)
    (run / "broker_state.json").write_text('{"schema":"paper"}\n', encoding="utf-8")
    (source / "kill_switch").write_text("HALT_NEW_ORDERS\n", encoding="utf-8")
    archive = tmp_path / "archive"
    backup_paper_state(source, archive)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["files"]}
    assert "kill_switch" in paths
    assert "metadata/paper/run-1/broker_state.json" in paths
    restored = tmp_path / "restored"
    restore_paper_state(archive, restored)
    kill = restored / "kill_switch"
    ledger = restored / "metadata" / "paper" / "run-1" / "broker_state.json"
    assert kill.read_text(encoding="utf-8") == "HALT_NEW_ORDERS\n"
    assert ledger.read_text(encoding="utf-8") == '{"schema":"paper"}\n'
    kill.write_text("ENABLED\n", encoding="utf-8")
    with pytest.raises(PaperBackupError, match="dest_differs"):
        restore_paper_state(archive, restored)
    assert kill.read_text(encoding="utf-8") == "ENABLED\n"
    shutil.rmtree(restored)
    restore_paper_state(archive, restored)
    assert kill.read_text(encoding="utf-8") == "HALT_NEW_ORDERS\n"
    assert ledger.read_text(encoding="utf-8") == '{"schema":"paper"}\n'
