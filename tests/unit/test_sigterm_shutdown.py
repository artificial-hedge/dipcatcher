"""Startup refusal and SIGTERM close for long-running entrypoints.

SYNTHETIC: the paper case uses a synthetic bar panel. Handlers are invoked
directly so the checks do not depend on the OS delivering SIGTERM.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fx1.interactive import actions, concierge, shell
from fx1.serve.local_engine import LocalWeightsEngine, main, serve
from quant_fund.config.loader import load_config
from quant_fund.observe import serve as metrics_serve
from quant_fund.paper.loop import run_paper_loop

pytestmark = pytest.mark.synthetic


@pytest.fixture
def saved_sigterm():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        yield previous
    finally:
        signal.signal(signal.SIGTERM, previous)


def _fire_sigterm() -> None:
    handler = signal.getsignal(signal.SIGTERM)
    if not callable(handler):
        raise AssertionError("SIGTERM handler was not installed")
    handler(signal.SIGTERM, None)


def _mini_cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    return cfg


def _bars(n: int = 6) -> pl.DataFrame:
    rows = []
    for day in range(n):
        ts = datetime(2024, 1, 1 + day, tzinfo=UTC)
        for sid, px in (("A", 100.0 + day), ("B", 50.0)):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": ts,
                    "open": px,
                    "close": px,
                    "close_total_return": px,
                    "volume": 1e6,
                    "adv": 1e8,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def test_local_engine_sigterm_closes_the_socket(
    monkeypatch: pytest.MonkeyPatch, saved_sigterm: object
) -> None:
    closed: list[bool] = []

    class _FakeServer:
        def __init__(self, address: tuple[str, int], handler: type[object]) -> None:
            del address, handler
            self.engine: object | None = None

        def serve_forever(self) -> None:
            _fire_sigterm()

        def server_close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("fx1.serve.local_engine._EngineHTTPServer", _FakeServer)
    code = serve(object(), "127.0.0.1", 0, watch_orphan=False)  # type: ignore[arg-type]
    assert code == 0
    assert closed == [True]
    assert signal.getsignal(signal.SIGTERM) is saved_sigterm


def test_local_engine_refuses_a_missing_checkpoint_before_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _bound(*args: object, **kwargs: object) -> int:
        raise AssertionError("serve must not bind")

    monkeypatch.setattr("fx1.serve.local_engine.serve", _bound)
    with pytest.raises(SystemExit) as missing:
        main(["--checkpoint-dir", str(tmp_path / "absent"), "--port", "9"])
    assert missing.value.code == 2
    with pytest.raises(FileNotFoundError, match="model card"):
        main(["--checkpoint-dir", str(tmp_path), "--port", "9"])


def test_unreadable_weights_manifest_refuses_to_serve(tmp_path: Path) -> None:
    manifest = tmp_path / "weights.manifest.json"
    manifest.write_bytes(b"\xff")
    with pytest.raises(RuntimeError, match="refusing to serve"):
        LocalWeightsEngine._load_manifest(tmp_path)
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to serve"):
        LocalWeightsEngine._load_manifest(tmp_path)
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="no weights manifest"):
        LocalWeightsEngine._load_manifest(tmp_path)


def test_paper_loop_sigterm_flushes_broker_state(tmp_path: Path, saved_sigterm: object) -> None:
    calls = {"n": 0}

    def champion_fn(dt: datetime, config: object) -> dict[str, float]:
        del dt, config
        calls["n"] += 1
        if calls["n"] == 1:
            _fire_sigterm()
        return {"A": 0.1, "B": -0.05}

    result = run_paper_loop(
        _bars(),
        _mini_cfg(tmp_path),
        champion_fn=champion_fn,
        initial_nav=100_000.0,
        max_steps=3,
        run_id="sigterm-flush",
        prefer_latest=False,
    )
    state_path = Path(result.paths["broker_state"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert result.metrics["n_steps_this_run"] == 1
    assert state["step"] == 1
    assert state_path.is_file()
    assert signal.getsignal(signal.SIGTERM) is saved_sigterm


def test_paper_resume_refuses_a_missing_broker_state(tmp_path: Path, saved_sigterm: object) -> None:
    with pytest.raises(FileNotFoundError, match="no broker_state.json"):
        run_paper_loop(
            _bars(2),
            _mini_cfg(tmp_path),
            champion_fn=lambda dt, config: {"A": 0.1},
            resume=True,
            run_id="missing-state",
            max_steps=1,
        )
    assert signal.getsignal(signal.SIGTERM) is saved_sigterm


def test_concierge_sigterm_joins_the_background_job(
    monkeypatch: pytest.MonkeyPatch, saved_sigterm: object
) -> None:
    class _Fake:
        model = "fx1"

        def __init__(self) -> None:
            self.shutdowns = 0

        def shutdown(self, timeout_s: float = 5.0) -> bool:
            del timeout_s
            self.shutdowns += 1
            return True

        def tick(self) -> None:
            return None

        def handle_line(self, line: str) -> bool:
            del line
            _fire_sigterm()
            return False

    fake = _Fake()
    monkeypatch.setattr(concierge.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(concierge.profiles, "resolve_endpoint", lambda model: None)

    concierge.run_console(
        concierge_factory=lambda: fake,
        input_fn=lambda prompt: "hello",
        output_fn=lambda *args, **kwargs: None,
    )
    assert fake.shutdowns == 1
    assert signal.getsignal(signal.SIGTERM) is saved_sigterm


def test_chat_sigterm_returns_cleanly(
    monkeypatch: pytest.MonkeyPatch, saved_sigterm: object
) -> None:
    monkeypatch.setattr(
        actions.profiles,
        "resolve_endpoint",
        lambda model: ("test-key", "http://127.0.0.1:9"),
    )

    def _input(prompt: str) -> str:
        del prompt
        _fire_sigterm()
        raise AssertionError("SIGTERM should have interrupted the read")

    code = actions.chat(
        "fx1",
        backend_factory=lambda **kwargs: object(),  # type: ignore[arg-type,return-value]
        input_fn=_input,
        output_fn=lambda *args, **kwargs: None,
        animated=False,
    )
    assert code == 0
    assert signal.getsignal(signal.SIGTERM) is saved_sigterm


def test_shell_sigterm_returns_cleanly(
    monkeypatch: pytest.MonkeyPatch, saved_sigterm: object
) -> None:
    def _cmdloop(self: object, intro: str | None = None) -> None:
        del self, intro
        _fire_sigterm()

    monkeypatch.setattr(shell.FxiShell, "cmdloop", _cmdloop)
    shell.run_shell(output_fn=lambda *args, **kwargs: None)
    assert signal.getsignal(signal.SIGTERM) is saved_sigterm


def test_metrics_server_closes_on_interpreter_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Callable[..., object]] = []
    real_register = metrics_serve.atexit.register

    def _spy(func: Callable[..., object], *args: object, **kwargs: object) -> object:
        recorded.append(func)
        return real_register(func, *args, **kwargs)

    monkeypatch.setattr(metrics_serve.atexit, "register", _spy)
    metrics_serve.start_metrics_server(0)
    try:
        assert recorded == [metrics_serve.stop_metrics_server]
    finally:
        metrics_serve.stop_metrics_server()
