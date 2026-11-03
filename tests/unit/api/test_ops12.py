"""Log retention, off-host backup, disk pressure, and the canary roll."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest
import yaml

from quant_fund.api.json_logging import append_audit_line, emit_service_event
from quant_fund.api.secret_files import clear_loaded_secrets
from quant_fund.api.service_config import ENV_CONTRACT

_ROOT = Path(__file__).resolve().parents[3]
_KEY = "planted-test-key-value-0123456789"
_OTHER = "metrics-test-key-0123456789"
_STAMP = "20261103T031500Z"
_CLEARED = (
    "QUANT_API_KEY",
    "RESEARCH_API_KEY",
    "DIPCATCHER_PROFILE",
    "DIPCATCHER_OBSERVE",
    "DIPCATCHER_ALERT_WEBHOOK_SECRET",
    "DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE",
    "DIPCATCHER_ALERT_WEBHOOK_URL",
    "DIPCATCHER_QUANT_API_KEY_FILE",
    "DIPCATCHER_RESEARCH_API_KEY_FILE",
    "DIPCATCHER_RATE_LIMIT_RPS",
    "DIPCATCHER_LAB_DATA",
    "DIPCATCHER_RESEARCH_DATA",
    "DIPCATCHER_TLS_DIR",
    "DIPCATCHER_KILL_SWITCH_PATH",
    "DIPCATCHER_AUDIT_LOG_PATH",
    "DIPCATCHER_AUDIT_LOG_MAX_BYTES",
    "DIPCATCHER_AUDIT_LOG_MAX_FILES",
    "DIPCATCHER_LOG_MAX_BYTES",
    "DIPCATCHER_LOG_MAX_AGE_S",
    "DIPCATCHER_LOG_MAX_FILES",
    "DIPCATCHER_APP_LOG_MAX_BYTES",
    "DIPCATCHER_PROXY_LOG_MAX_BYTES",
    "DIPCATCHER_PROXY_LOG_MAX_AGE_S",
    "DIPCATCHER_AUDIT_LOG_MAX_AGE_S",
    "DIPCATCHER_BACKUP_COPY",
    "DIPCATCHER_BACKUP_RSYNC_TARGET",
    "DIPCATCHER_BACKUP_S3_URI",
    "DIPCATCHER_LOG_DIR",
    "DIPCATCHER_LOG_MAX_SIZE",
    "DIPCATCHER_LOG_MAX_FILE",
)


class _Done:
    def __init__(self, code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = code
        self.stdout = stdout
        self.stderr = stderr


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _retain():
    return _load(_ROOT / "scripts" / "log_retain.py", "dipcatcher_log_retain_ops12")


def _ship():
    return _load(_ROOT / "scripts" / "prod_backup_ship.py", "dipcatcher_prod_backup_ship_ops12")


def _canary():
    return _load(_ROOT / "scripts" / "prod_canary.py", "dipcatcher_prod_canary_ops12")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    clear_loaded_secrets()
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)
    yield
    clear_loaded_secrets()


@pytest.fixture
def paper() -> tuple[Path, Path]:
    base = Path("/tmp") / f"dipcatcher-ops12-{uuid.uuid4().hex}"
    source = base / "lab"
    root = base / "backups"
    source.mkdir(parents=True)
    (source / "broker_state.json").write_text('{"synthetic": true}\n', encoding="utf-8")
    try:
        yield source, root
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _policies(max_bytes: int, max_age_s: int, max_files: int) -> dict:
    policy = _retain().Policy(max_bytes, max_age_s, max_files)
    return {stream: policy for stream in ("app", "proxy", "audit")}


def test_retention_rotates_on_size_and_keeps_the_ledger(tmp_path: Path) -> None:
    module = _retain()
    for stream in ("app", "proxy", "audit"):
        (tmp_path / stream).mkdir()
    active = tmp_path / "app" / "service.log"
    active.write_bytes(b"x" * 50)
    ledger = tmp_path / "audit" / "entries.jsonl"
    ledger.write_bytes(b"y" * 80)
    numbered = tmp_path / "audit" / "checkpoints.jsonl.2"
    numbered.write_text("kept", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "proxy" / "proxy.log"
    link.symlink_to(outside)
    old = time.time() - 10
    os.utime(ledger, (old, old))
    os.utime(numbered, (old, old))
    report = module.retain_tree(tmp_path, _policies(10, 86400, 5), time.time())
    assert report["ok"] is True
    assert not active.exists()
    assert (tmp_path / "app" / "service.log.1").read_bytes() == b"x" * 50
    assert ledger.read_bytes() == b"y" * 80
    assert numbered.read_text(encoding="utf-8") == "kept"
    assert link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "secret"


def test_retention_drops_rotated_copies_for_age_and_count(tmp_path: Path) -> None:
    module = _retain()
    for stream in ("app", "proxy", "audit"):
        (tmp_path / stream).mkdir()
    aged = tmp_path / "proxy" / "access.log.1"
    aged.write_text("old", encoding="utf-8")
    os.utime(aged, (1, 1))
    extra = tmp_path / "app" / "service.log.6"
    extra.write_text("extra", encoding="utf-8")
    now = time.time()
    os.utime(extra, (now, now))
    report = module.retain_tree(tmp_path, _policies(10**9, 10, 5), now)
    assert report["ok"] is True
    assert not aged.exists()
    assert not extra.exists()


def test_log_policies_reject_junk_and_honor_a_stream() -> None:
    module = _retain()
    assert module.policies_from_env({"DIPCATCHER_LOG_MAX_BYTES": "nope"}) is None
    assert module.policies_from_env({"DIPCATCHER_LOG_MAX_FILES": "0"}) is None
    found = module.policies_from_env(
        {"DIPCATCHER_APP_LOG_MAX_BYTES": "100", "DIPCATCHER_PROXY_LOG_MAX_AGE_S": "3"}
    )
    assert found is not None
    assert found["app"].max_bytes == 100
    assert found["proxy"].max_age_s == 3
    assert found["audit"].max_bytes == module.DEFAULT_MAX_BYTES
    assert module.sealed_name("entries.jsonl")
    assert module.sealed_name("checkpoints.jsonl.4")
    assert not module.sealed_name("service.log")


def test_docker_prune_deletes_rotated_logs_only(tmp_path: Path) -> None:
    module = _retain()
    hexid = "a" * 64
    directory = tmp_path / "containers" / hexid
    directory.mkdir(parents=True)
    live = directory / f"{hexid}-json.log"
    live.write_text("live", encoding="utf-8")
    rotated = directory / f"{hexid}-json.log.1"
    rotated.write_text("old", encoding="utf-8")
    os.utime(rotated, (1, 1))
    assert module.docker_log_dir_ok(live)
    assert not module.docker_log_dir_ok(tmp_path / "service.log")
    report = module.prune_docker_logs([str(live)], module.Policy(10**9, 10, 5), 100.0)
    assert report["ok"] is True
    assert live.read_text(encoding="utf-8") == "live"
    assert not rotated.exists()
    items = [
        module.Item(live.name, 4, 100.0, False),
        module.Item(rotated.name, 3, 100.0, False),
    ]
    planned = module.plan_rotated_only(items, module.Policy(10**9, 10, 5), 100.0)
    assert {item.name: item.op for item in planned}[live.name] == "keep"
    assert module.stream_for_container("/dipcatcher-ops7-proxy-1") == "proxy"
    assert module.stream_for_container("/dipcatcher-ops7-lab-a-1") == "app"
    assert not module.project_ok("has space")


def test_audit_file_redacts_and_rotates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "lab-a.jsonl"
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_PATH", str(path))
    monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_MAX_BYTES", "80")
    monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_MAX_FILES", "2")
    emit_service_event("request", detail=_KEY)
    text = path.read_text(encoding="utf-8")
    assert _KEY not in text
    assert "[redacted]" in text
    append_audit_line("b" * 70)
    assert path.is_file()
    assert (tmp_path / "lab-a.jsonl.1").is_file()
    link = tmp_path / "linked.jsonl"
    link.symlink_to(path)
    monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_PATH", str(link))
    before = path.read_text(encoding="utf-8")
    append_audit_line("ignored")
    assert path.read_text(encoding="utf-8") == before
    monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_PATH", "relative.jsonl")
    append_audit_line("ignored")
    assert not Path("relative.jsonl").exists()


def test_new_log_and_backup_names_stay_out_of_the_contract() -> None:
    for name in (
        "DIPCATCHER_AUDIT_LOG_PATH",
        "DIPCATCHER_LOG_MAX_BYTES",
        "DIPCATCHER_BACKUP_COPY",
        "DIPCATCHER_BACKUP_RSYNC_TARGET",
        "DIPCATCHER_BACKUP_S3_URI",
        "DIPCATCHER_LOG_DIR",
    ):
        assert name not in ENV_CONTRACT


def test_backup_ship_verifies_before_copy(paper: tuple[Path, Path]) -> None:
    source, root = paper
    module = _ship()
    calls: list[list[str]] = []

    def _run(argv: list[str]) -> _Done:
        calls.append(list(argv))
        return _Done()

    report = module.ship(source, root, {}, _run, _STAMP)
    assert report["ok"] is True
    assert report["verified"] is True
    assert report["copy"] == "none"
    assert calls == []
    manifest = root / _STAMP / "manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["manifest_sha256"] == digest
    payload = root / _STAMP / "payload" / "broker_state.json"
    payload.write_text("tampered\n", encoding="utf-8")
    checked = module.verify_archive(root / _STAMP)
    assert checked == {"error": "hash_mismatch", "ok": False, "verified": False}

    def _refuse(_archive: Path) -> dict[str, object]:
        return {"error": "hash_mismatch", "ok": False, "verified": False}

    again = root.parent / "again"
    refused = module.ship(
        source,
        again,
        {
            "DIPCATCHER_BACKUP_COPY": "rsync",
            "DIPCATCHER_BACKUP_RSYNC_TARGET": "/var/backups/dipcatcher",
        },
        _run,
        _STAMP,
        verify=_refuse,
    )
    assert refused["ok"] is False
    assert calls == []


def test_backup_ship_rsync_and_s3_argv(paper: tuple[Path, Path]) -> None:
    source, root = paper
    module = _ship()
    calls: list[list[str]] = []

    def _rsync(argv: list[str]) -> _Done:
        calls.append(list(argv))
        if "--dry-run" in argv:
            return _Done(0, "")
        return _Done()

    report = module.ship(
        source,
        root,
        {
            "DIPCATCHER_BACKUP_COPY": "rsync",
            "DIPCATCHER_BACKUP_RSYNC_TARGET": "backup@backup.example:/var/backups/dipcatcher",
        },
        _rsync,
        _STAMP,
    )
    assert report["ok"] is True
    assert calls[0][:3] == ["rsync", "-a", "--checksum"]
    assert calls[0][-1] == f"backup@backup.example:/var/backups/dipcatcher/{_STAMP}/"
    assert "--dry-run" in calls[1]
    assert "--itemize-changes" in calls[1]
    drifted: list[list[str]] = []

    def _drift(argv: list[str]) -> _Done:
        drifted.append(list(argv))
        if "--dry-run" in argv:
            return _Done(0, ">f.st.... manifest.json\n")
        return _Done()

    drift_root = root.parent / "drift"
    drift = module.ship(
        source,
        drift_root,
        {
            "DIPCATCHER_BACKUP_COPY": "rsync",
            "DIPCATCHER_BACKUP_RSYNC_TARGET": "/var/backups/dipcatcher",
        },
        _drift,
        _STAMP,
    )
    assert drift["ok"] is False
    assert drift["error"] == "rsync drift"
    s3_calls: list[list[str]] = []

    def _s3(argv: list[str]) -> _Done:
        s3_calls.append(list(argv))
        if argv[:3] == ["aws", "s3", "cp"]:
            text = (root.parent / "s3" / _STAMP / "manifest.json").read_text(encoding="utf-8")
            return _Done(0, text)
        return _Done()

    copied = module.ship(
        source,
        root.parent / "s3",
        {"DIPCATCHER_BACKUP_COPY": "s3", "DIPCATCHER_BACKUP_S3_URI": "s3://bucket/dipcatcher"},
        _s3,
        _STAMP,
    )
    assert copied["ok"] is True
    assert s3_calls[0][:4] == ["aws", "s3", "sync", str(root.parent / "s3" / _STAMP)]
    assert s3_calls[1] == [
        "aws",
        "s3",
        "cp",
        f"s3://bucket/dipcatcher/{_STAMP}/manifest.json",
        "-",
    ]


def test_backup_targets_and_paths_are_refused(paper: tuple[Path, Path]) -> None:
    source, root = paper
    module = _ship()
    assert module.rsync_target_ok("/var/backups/dipcatcher")
    assert module.rsync_target_ok("backup@backup.example:/var/backups/dipcatcher")
    assert not module.rsync_target_ok("user:pass@host:/tmp/dipcatcher")
    assert not module.rsync_target_ok("-rf")
    assert not module.rsync_target_ok("user@host:relative")
    assert not module.rsync_target_ok("/tmp/dipcatcher/../etc")
    assert module.s3_uri_ok("s3://bucket/dipcatcher")
    assert not module.s3_uri_ok("s3://bucket/../dipcatcher")
    assert not module.s3_uri_ok("s3://user@bucket/dipcatcher")
    assert module.locate(Path("/tmp")) is None
    assert module.locate(Path("/var/backups")) is None
    assert module.locate(source) == source.resolve()
    calls: list[list[str]] = []
    refused = module.ship(
        source,
        root,
        {"DIPCATCHER_BACKUP_COPY": "both"},
        lambda argv: calls.append(list(argv)),
        _STAMP,
    )
    assert refused["error"] == "copy"
    assert calls == []
    assert not root.exists()
    (root).mkdir()
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    nested = module.ship(source, root, {}, lambda argv: calls.append(list(argv)), _STAMP)
    assert nested["error"] == "archive_root"


def test_canary_rolls_or_aborts() -> None:
    module = _canary()
    assert module.decide(True) == "roll"
    assert module.decide(False) == "abort"
    assert module.smoke_passed({"health": 200, "small": 405, "large": 413})
    assert not module.smoke_passed({"health": 200, "small": 413, "large": 413})
    assert not module.smoke_passed({"health": 500, "small": 200, "large": 413})
    env = {
        "QUANT_API_KEY": _KEY,
        "RESEARCH_API_KEY": _OTHER,
        "SUDO": "",
    }
    calls: list[list[str]] = []

    def _run(argv: list[str]) -> _Done:
        calls.append(list(argv))
        if "inspect" in argv:
            return _Done(0, "healthy\n")
        if "exec" in argv:
            return _Done(0, json.dumps({"health": 200, "small": 405, "large": 413}))
        return _Done()

    report = module.execute(env, _run, sleep=lambda _seconds: None, attempts=2)
    assert report["ok"] is True
    assert report["action"] == "roll"
    assert report["removed"] is True
    assert calls[0][0] == "docker"
    assert "--profile" in calls[0] and "canary" in calls[0] and "--no-deps" in calls[0]
    assert "inspect" in calls[1]
    assert "exec" in calls[2]
    assert "--force-recreate" in calls[3]
    assert calls[3][-5:] == ["lab-a", "lab-b", "research-a", "research-b", "proxy"]
    assert "lab-canary" not in calls[3]
    assert "rm" in calls[4] and "lab-canary" in calls[4]
    aborted: list[list[str]] = []

    def _abort(argv: list[str]) -> _Done:
        aborted.append(list(argv))
        if "inspect" in argv:
            return _Done(0, "healthy\n")
        if "exec" in argv:
            return _Done(0, json.dumps({"health": 200, "small": 200, "large": 200}))
        return _Done()

    stopped = module.execute(env, _abort, sleep=lambda _seconds: None, attempts=2)
    assert stopped["ok"] is False
    assert stopped["action"] == "abort"
    assert not any("--force-recreate" in item for item in aborted)
    assert any("rm" in item and "lab-canary" in item for item in aborted)
    missed: list[object] = []
    missing = module.execute({}, lambda argv: missed.append(argv))
    assert missing == {"error": module._KEY_ERROR, "ok": False}
    assert missed == []
    preserved: list[list[str]] = []

    def _sudo(argv: list[str]) -> _Done:
        preserved.append(list(argv))
        return _Done(1, "")

    sudo_env = {
        "QUANT_API_KEY": _KEY,
        "RESEARCH_API_KEY": _OTHER,
        "SUDO": "sudo",
        "DIPCATCHER_COMPOSE_PRESERVE": "DIPCATCHER_LAB_DATA",
    }
    failed = module.execute(sudo_env, _sudo, attempts=1)
    assert failed["ok"] is False
    assert failed["removed"] is False
    assert "--preserve-env=DIPCATCHER_LAB_DATA" in preserved[0]


def test_compose_units_and_host_guide() -> None:
    compose_text = (_ROOT / "deploy" / "compose.prod.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    canary = compose["services"]["lab-canary"]
    assert canary["profiles"] == ["canary"]
    assert canary["environment"]["DIPCATCHER_WRITE_OWNER"] == "0"
    assert canary["command"] == ["dipcatcher", "api"]
    assert compose["services"]["lab-a"]["logging"]["driver"] == "json-file"
    assert "10m" in compose["services"]["lab-a"]["logging"]["options"]["max-size"]
    assert compose["services"]["node-exporter"]["image"] == "prom/node-exporter:v1.8.2"
    assert "ports" not in compose["services"]["node-exporter"]
    assert "lab-canary" not in (_ROOT / "deploy" / "proxy" / "Caddyfile").read_text(
        encoding="utf-8"
    )
    assert compose_text.count("/app/data:ro") == 1
    assert "0.0.0.0:" not in compose_text
    alerts = (_ROOT / "deploy" / "observability" / "alerts.yml").read_text(encoding="utf-8")
    assert "DipcatcherDiskPressure" in alerts
    assert "node_filesystem_avail_bytes" in alerts
    assert "0.15" in alerts
    prometheus = yaml.safe_load(
        (_ROOT / "deploy" / "observability" / "prometheus.yml").read_text(encoding="utf-8")
    )
    assert prometheus["rule_files"] == ["alerts.yml", "slo_rules.yml"]
    jobs = [item["job_name"] for item in prometheus["scrape_configs"]]
    assert "node" in jobs
    unit = (_ROOT / "deploy" / "systemd" / "dipcatcher-prod-backup.service").read_text(
        encoding="utf-8"
    )
    timer = (_ROOT / "deploy" / "systemd" / "dipcatcher-prod-backup.timer").read_text(
        encoding="utf-8"
    )
    log_unit = (_ROOT / "deploy" / "systemd" / "dipcatcher-log-retain.service").read_text(
        encoding="utf-8"
    )
    log_timer = (_ROOT / "deploy" / "systemd" / "dipcatcher-log-retain.timer").read_text(
        encoding="utf-8"
    )
    assert "ExecStart=/opt/dipcatcher/.venv/bin/python" in unit
    assert "prod_backup_ship.py" in unit
    assert "OnCalendar=*-*-* 03:15:00" in timer
    assert "Persistent=true" in timer
    assert "log_retain.py" in log_unit
    assert "OnCalendar=hourly" in log_timer
    cron = (_ROOT / "deploy" / "cron" / "dipcatcher-prod-backup").read_text(encoding="utf-8")
    assert "prod_backup_ship.py" in cron
    assert "log_retain.py" in (_ROOT / "deploy" / "cron" / "dipcatcher-log-retain").read_text(
        encoding="utf-8"
    )
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "prod-up: prod-certs prod-data prod-preflight prod-build" in prod
    for name in ("prod-canary", "prod-log-retain", "prod-backup-ship"):
        assert f"{name}:" in prod
        assert f"{name}:" not in makefile
    guide = (_ROOT / "docs" / "DEPLOY_REAL_HOST.md").read_text(encoding="utf-8")
    for snippet in (
        "apt-get install -y docker-ce",
        "ufw allow 443/tcp",
        "ufw allow 8443/tcp",
        "ufw allow 8444/tcp",
        "prod-secrets",
        "prod-smoke",
        "prod-up",
        "dipcatcher-prod-backup.timer",
        "ACME",
        "dev.crt",
    ):
        assert snippet in guide
    assert "BEGIN PRIVATE KEY" not in guide
    assert "ufw allow 22" not in guide
    example = (_ROOT / "deploy" / "prod.env.example").read_text(encoding="utf-8")
    assert "DIPCATCHER_LOG_MAX_BYTES" in example
    assert "DIPCATCHER_BACKUP_COPY" in example
    assert "API_KEY=" not in example
