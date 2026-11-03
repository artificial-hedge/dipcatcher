"""Drain debounce, entrypoint writability, and the container entry contract."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_fund.utils.operator_kill import ENV_PATH, DrainLatch, write_operator_kill

_ROOT = Path(__file__).resolve().parents[3]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _entrypoint():
    path = _ROOT / "docker" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("dipcatcher_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lines(text: str) -> list[dict[str, str]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_drain_clears_after_debounce(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "HALT_NEW_ORDERS")
    clock = _Clock()
    latch = DrainLatch(debounce_s=1.0, clock=clock)
    event = threading.Event()
    assert latch.poll(event, path) == "latched"
    assert event.is_set()
    write_operator_kill(path, "ENABLED")
    assert latch.poll(event, path) == "waiting"
    clock.now = 0.9
    assert latch.poll(event, path) == "waiting"
    assert event.is_set()
    clock.now = 1.0
    assert latch.poll(event, path) == "cleared"
    assert not event.is_set()
    logged = _lines(capsys.readouterr().err)
    assert logged[0]["event"] == "operator_kill_drain_latched"
    assert logged[0]["code"] == "kill_switch:HALT_NEW_ORDERS"
    assert logged[-1]["event"] == "operator_kill_drain_cleared"
    assert "HALT_NEW_ORDERS" not in logged[-1].get("code", "")


def test_drain_flicker_resets_the_timer(tmp_path: Path) -> None:
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "HALT_NEW_ORDERS")
    clock = _Clock()
    latch = DrainLatch(debounce_s=1.0, clock=clock)
    event = threading.Event()
    assert latch.poll(event, path) == "latched"
    write_operator_kill(path, "ENABLED")
    assert latch.poll(event, path) == "waiting"
    clock.now = 0.5
    write_operator_kill(path, "CANCEL_OPEN_ORDERS")
    assert latch.poll(event, path) == "held"
    write_operator_kill(path, "ENABLED")
    clock.now = 10.0
    assert latch.poll(event, path) == "waiting"
    assert event.is_set()
    clock.now = 11.0
    assert latch.poll(event, path) == "cleared"
    assert not event.is_set()


def test_zero_debounce_clears_on_the_same_poll(tmp_path: Path) -> None:
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "HALT_NEW_ORDERS")
    latch = DrainLatch(debounce_s=0.0, clock=_Clock())
    event = threading.Event()
    assert latch.poll(event, path) == "latched"
    write_operator_kill(path, "ENABLED")
    assert latch.poll(event, path) == "cleared"
    assert not event.is_set()


def test_invalid_debounce_never_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DIPCATCHER_KILL_DRAIN_DEBOUNCE_S", "nope")
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "HALT_NEW_ORDERS")
    clock = _Clock()
    latch = DrainLatch.from_env(clock=clock)
    event = threading.Event()
    assert latch.poll(event, path) == "latched"
    write_operator_kill(path, "ENABLED")
    clock.now = 10_000.0
    assert latch.poll(event, path) == "waiting"
    assert event.is_set()
    err = capsys.readouterr().err
    assert "operator_kill_drain_debounce_invalid" in err
    assert "nope" not in err


def test_symlink_stays_latched(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.write_text("ENABLED\n", encoding="utf-8")
    link = tmp_path / "kill_switch"
    link.symlink_to(target)
    clock = _Clock()
    latch = DrainLatch(debounce_s=0.0, clock=clock)
    event = threading.Event()
    assert latch.poll(event, link) == "latched"
    clock.now = 5.0
    assert latch.poll(event, link) == "held"
    assert event.is_set()


def test_http_drain_is_not_cleared_by_an_enabled_file(tmp_path: Path) -> None:
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "ENABLED")
    event = threading.Event()
    event.set()
    latch = DrainLatch(debounce_s=0.0, clock=_Clock())
    assert latch.poll(event, path) == "open"
    assert event.is_set()
    write_operator_kill(path, "HALT_NEW_ORDERS")
    assert latch.poll(event, path) == "held"
    write_operator_kill(path, "ENABLED")
    assert latch.poll(event, path) == "open"
    assert event.is_set()


def test_harness_wrapper_clears_its_latch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    from fx1.serve.kill_drain import serve_until_signal

    monkeypatch.setenv("DIPCATCHER_KILL_DRAIN_DEBOUNCE_S", "0")
    path = tmp_path / "kill_switch"
    write_operator_kill(path, "ENABLED")
    monkeypatch.setenv(ENV_PATH, str(path))
    event = threading.Event()
    app = SimpleNamespace(state=SimpleNamespace(metrics=SimpleNamespace(draining=event)))

    def fake_run(*_args: object, **_kwargs: object) -> None:
        write_operator_kill(path, "HALT_NEW_ORDERS")
        assert event.wait(2)
        write_operator_kill(path, "ENABLED")
        deadline = time.monotonic() + 2
        while event.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not event.is_set()

    monkeypatch.setattr(uvicorn, "run", fake_run)
    serve_until_signal(app, host="127.0.0.1", port=9)
    err = capsys.readouterr().err
    assert "operator_kill_drain_latched" in err
    assert "operator_kill_drain_cleared" in err


def test_entrypoint_rejects_symlink_missing_and_unwritable(tmp_path: Path) -> None:
    entry = _entrypoint()
    missing = tmp_path / "missing"
    message = entry.check_data_dir(str(missing), require_write=True)
    assert message is not None
    assert "is not a directory" in message
    assert "uid 10001" in message
    link = tmp_path / "link"
    target = tmp_path / "target"
    target.mkdir()
    link.symlink_to(target)
    linked = entry.check_data_dir(str(link), require_write=False)
    assert linked is not None
    assert "must not be a symlink" in linked
    writable = entry.check_data_dir(str(target), require_write=True)
    assert writable is None
    assert list(target.glob(".dipcatcher-write-probe-*")) == []


def test_entrypoint_write_failure_names_the_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entrypoint()
    target = tmp_path / "data"
    target.mkdir()

    def _denied(*_args: object, **_kwargs: object) -> int:
        raise PermissionError("denied")

    monkeypatch.setattr(entry.os, "open", _denied)
    message = entry.check_data_dir(str(target), require_write=True)
    assert message == (
        f"data directory {target} is not writable by uid {os.geteuid()}; "
        "the dipcatcher user is uid 10001"
    )
    assert "denied" not in (message or "")


def test_entrypoint_mode_blocks_non_root(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode bits")
    entry = _entrypoint()
    target = tmp_path / "locked"
    target.mkdir()
    os.chmod(target, 0o555)
    try:
        message = entry.check_data_dir(str(target), require_write=True)
        assert message is not None
        assert "not writable" in message
        assert "uid 10001" in message
        assert entry.check_data_dir(str(target), require_write=False) is None
    finally:
        os.chmod(target, 0o755)


def test_entrypoint_main_fails_before_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = _entrypoint()
    monkeypatch.setenv("DIPCATCHER_DATA_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("DIPCATCHER_REQUIRE_DATA_WRITE", "1")
    assert entry.main(["entrypoint", "dipcatcher", "api"]) == 1
    assert "uid 10001" in capsys.readouterr().err


def test_container_entrypoints_keep_cmds() -> None:
    lab = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    research = (_ROOT / "docker" / "research-api.Dockerfile").read_text(encoding="utf-8")
    entry = 'ENTRYPOINT ["/app/.venv/bin/python", "/app/entrypoint.py"]'
    assert entry in lab
    assert entry in research
    assert "DIPCATCHER_REQUIRE_DATA_WRITE=1" in lab
    assert "DIPCATCHER_REQUIRE_DATA_WRITE=0" in research
    assert 'CMD ["dipcatcher", "api"]' in lab
    assert 'CMD ["python", "-m", "quant_fund.api.research_api"]' in research
    assert "/app/healthcheck.py" in lab
    assert "docker/entrypoint.py" in lab
    assert "docker/entrypoint.py" in research
