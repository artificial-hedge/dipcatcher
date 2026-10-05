"""Bounded startup and socket ownership for the loopback HTTP quickstart."""

from __future__ import annotations

import importlib.util
import json
import socket
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi import FastAPI


@pytest.fixture
def quickstart() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples/fx1_quickstart_http.py"
    spec = importlib.util.spec_from_file_location("fx1_quickstart_http", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_startup_failure_is_reported_and_socket_closed(
    quickstart: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets_seen: list[socket.socket] = []

    class FailedServer:
        started = False
        should_exit = False
        force_exit = False

        def __init__(self, config: Any) -> None:
            pass

        def run(self, *, sockets: list[socket.socket]) -> None:
            sockets_seen.extend(sockets)
            raise RuntimeError("synthetic startup failure")

    monkeypatch.setattr(quickstart.uvicorn, "Server", FailedServer)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="exited before startup") as caught:
        quickstart._serve(FastAPI(), startup_timeout_s=0.2)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert time.monotonic() - start < 3
    assert sockets_seen and all(s.fileno() == -1 for s in sockets_seen)


def test_stalled_startup_times_out_and_stops_worker(
    quickstart: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets_seen: list[socket.socket] = []
    stopped: list[bool] = []

    class StalledServer:
        started = False
        should_exit = False
        force_exit = False

        def __init__(self, config: Any) -> None:
            pass

        def run(self, *, sockets: list[socket.socket]) -> None:
            sockets_seen.extend(sockets)
            while not self.should_exit:
                time.sleep(0.001)
            stopped.append(True)

    monkeypatch.setattr(quickstart.uvicorn, "Server", StalledServer)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="startup timed out"):
        quickstart._serve(FastAPI(), startup_timeout_s=0.02)
    assert time.monotonic() - start < 3
    assert stopped == [True]
    assert sockets_seen and all(s.fileno() == -1 for s in sockets_seen)


def test_uvicorn_receives_the_reserved_socket(
    quickstart: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sockets_seen: list[socket.socket] = []
    reserved: list[bool] = []

    class RunningServer:
        started = False
        should_exit = False
        force_exit = False

        def __init__(self, config: Any) -> None:
            pass

        def run(self, *, sockets: list[socket.socket]) -> None:
            sockets_seen.extend(sockets)
            with socket.socket() as competitor, pytest.raises(OSError):
                competitor.bind(sockets[0].getsockname())
            reserved.append(True)
            self.started = True
            while not self.should_exit:
                time.sleep(0.001)

    monkeypatch.setattr(quickstart.uvicorn, "Server", RunningServer)
    server, worker, port = quickstart._serve(FastAPI(), startup_timeout_s=0.5)
    try:
        assert reserved == [True]
        assert sockets_seen[0].getsockname() == ("127.0.0.1", port)
    finally:
        server.should_exit = True
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert sockets_seen[0].fileno() == -1


def test_real_uvicorn_serves_and_shuts_down(quickstart: ModuleType) -> None:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    server, worker, port = quickstart._serve(app, startup_timeout_s=3)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}
    finally:
        server.should_exit = True
        worker.join(timeout=3)
    assert not worker.is_alive()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0.0])
def test_invalid_startup_timeout_is_rejected(quickstart: ModuleType, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        quickstart._serve(FastAPI(), startup_timeout_s=timeout)
