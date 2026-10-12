"""Planner ceilings and superpower transcript follow-ups.

``max-harness`` caps planned steps. ``max-families`` does not. An explicit
``max-seconds`` trims steps whose estimated runtime exceeds it. A typed
harness failure sets the turn status. A later ``why?`` still sees the goal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.flash.store import FlashStore
from fx1.interactive import profiles
from fx1.interactive.concierge import Concierge
from fx1.serve.backends import ToolCompletion
from tests.fx1.capabilities_fixture import sample_catalog_document


class _Result:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = False
        self.cancelled = False
        self.truncated = False


class _CatalogHarness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, name: str, extra_args: list[str] | None = None, **_kwargs: Any) -> _Result:
        args = list(extra_args or [])
        self.calls.append((name, args))
        if name == "capabilities":
            return _Result(0, json.dumps(sample_catalog_document()), "")
        if name == "families":
            return _Result(0, _families(args), "")
        return _Result(0, f"DATA_LABEL=SYNTHETIC\n{name} finished\n", "")


class _FixedHarness:
    def __init__(self, exit_code: int, stderr: str = "") -> None:
        self._exit = exit_code
        self._stderr = stderr

    def run(self, name: str, extra_args: list[str] | None = None, **_kwargs: Any) -> _Result:
        del name, extra_args
        return _Result(self._exit, "DATA_LABEL=SYNTHETIC\n", self._stderr)


class _Backend:
    def __init__(self) -> None:
        self.chat: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        del messages
        return "QLIKE on sealed shards. SYNTHETIC correctness check only."

    def complete_with_tools(self, messages: list[dict[str, Any]], **_kwargs: Any) -> ToolCompletion:
        self.chat.append(list(messages))
        return ToolCompletion(
            content="The lane followed the volatility wording.",
            tool_calls=None,
            finish_reason="stop",
        )


def _families(args: list[str]) -> str:
    goal = ""
    if "--select" in args:
        goal = args[args.index("--select") + 1].casefold()
    selected = [
        {"id": family_id, "executable": True, "qualification": "required"}
        for family_id in ("volatility", "conformal")
        if family_id in goal
    ]
    payload = {"schema": "dipcatcher.family_surface.v1", "selected": selected, "exclusions": []}
    return json.dumps(payload)


def _open(
    tmp_path: Path, harness: Any, backend: _Backend | None = None
) -> tuple[Concierge, list[str]]:
    out: list[str] = []
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    concierge = Concierge(
        store=store,
        harness=harness,
        backend_factory=(lambda **_kwargs: backend) if backend is not None else None,
        output_fn=out.append,
        animated=False,
    )
    return concierge, out


def _ran(harness: _CatalogHarness) -> list[str]:
    return [name for name, _args in harness.calls if name not in {"capabilities", "families"}]


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    return tmp_path


def test_max_families_does_not_cap_harness_steps(env: Path, tmp_path: Path) -> None:
    del env
    harness = _CatalogHarness()
    concierge, out = _open(tmp_path, harness, _Backend())
    profiles.set_endpoint("fx1", "test-key-abcdefgh", "http://127.0.0.1:9/v1")
    concierge.handle_line("/budget max-families 1")
    concierge.handle_line(
        "/superpower score volatility forecasts and evaluate calibration with a pit score --sync"
    )
    text = "\n".join(out)
    assert "max_families=1" in text
    assert "exceed max_steps=1" not in text
    assert _ran(harness) == ["calibration-eval", "vol-bench"] or _ran(harness) == [
        "vol-bench",
        "calibration-eval",
    ]
    family = [args for name, args in harness.calls if name == "families"]
    assert family == [] or "1" in family[0]


def test_max_harness_caps_planned_steps(env: Path, tmp_path: Path) -> None:
    del env
    harness = _CatalogHarness()
    concierge, out = _open(tmp_path, harness, _Backend())
    profiles.set_endpoint("fx1", "test-key-abcdefgh", "http://127.0.0.1:9/v1")
    concierge.handle_line("/budget max-harness 1")
    concierge.handle_line(
        "/superpower score volatility forecasts and evaluate calibration with a pit score --sync"
    )
    text = "\n".join(out)
    assert "exceed max_steps=1" in text
    assert len(_ran(harness)) == 1


def test_explicit_max_seconds_drops_steps_over_the_estimate(env: Path, tmp_path: Path) -> None:
    del env
    harness = _CatalogHarness()
    concierge, out = _open(tmp_path, harness, _Backend())
    profiles.set_endpoint("fx1", "test-key-abcdefgh", "http://127.0.0.1:9/v1")
    concierge.handle_line("/budget max-seconds 10")
    concierge.handle_line("/superpower score volatility forecasts with QLIKE --sync")
    text = "\n".join(out)
    assert "estimated runtime exceeds budget" in text
    assert "vol-bench" not in _ran(harness)
    assert "calibration-eval" not in _ran(harness)


def test_failed_typed_run_sets_error_status(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = _open(tmp_path, _FixedHarness(2, "doctor failed closed\n"))
    concierge.handle_line("/harness run doctor")
    assert concierge.last_status == "error"
    assert "DATA_LABEL=SYNTHETIC" in "\n".join(out)
    concierge.handle_line("/harness run not-a-lab")
    assert concierge.last_status == "error"
    assert "unknown_command" in "\n".join(out)


def test_why_after_superpower_includes_the_goal(env: Path, tmp_path: Path) -> None:
    del env
    backend = _Backend()
    harness = _CatalogHarness()
    concierge, _out = _open(tmp_path, harness, backend)
    profiles.set_endpoint("fx1", "test-key-abcdefgh", "http://127.0.0.1:9/v1")
    concierge.handle_line(
        "/superpower score volatility forecasts with QLIKE marker-qlike-771 --sync"
    )
    concierge.handle_line("why?")
    blob = json.dumps(backend.chat)
    assert "marker-qlike-771" in blob
    assert "vol-bench" in blob
