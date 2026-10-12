"""Helpers for assembled-product red-team tests. Not collected by pytest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.flash.store import FlashStore
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.usage_budget import UsageLimits
from fx1.serve.backends import ToolCompletion
from tests.fx1.test_concierge import FakeBackend, FakeHarnessResult, FakeResearcher

SECRET = "sk-" + ("b" * 24)
CANARY_TEXT = "intact"


def isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point config, sessions, and home at *tmp_path* and leave the real home alone."""
    home = tmp_path / "home"
    config = tmp_path / "fx1-config"
    home.mkdir()
    config.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("FX1_CONFIG_DIR", str(config))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.chdir(tmp_path)
    return config


def tool(call_id: str, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def completion(
    content: str | None,
    calls: tuple[dict[str, Any], ...] | None = None,
) -> ToolCompletion:
    return ToolCompletion(
        content=content,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
    )


class RecordingHarness:
    """Touches *canary* only if ``run`` is actually invoked."""

    def __init__(self, canary: Path) -> None:
        self.canary = canary
        self.calls: list[tuple[str, list[str]]] = []

    def run(
        self,
        name: str,
        extra_args: list[str] | None = None,
        **_kwargs: Any,
    ) -> FakeHarnessResult:
        args = [str(item) for item in (extra_args or [])]
        self.calls.append((name, args))
        self.canary.write_text("executed", encoding="utf-8")
        return FakeHarnessResult(0, "harness-ok\n")


def plant_canary(tmp_path: Path) -> Path:
    path = tmp_path / "canary.txt"
    path.write_text(CANARY_TEXT, encoding="utf-8")
    return path


def open_chat(
    tmp_path: Path,
    script: list[ToolCompletion],
    *,
    harness: Any | None = None,
    approvals: ApprovalGate | None = None,
    researcher: FakeResearcher | None = None,
    store: FlashStore | None = None,
) -> tuple[Concierge, list[str], Any]:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key-value", "https://models.example.test/v1")
    out: list[str] = []
    owned = harness if harness is not None else RecordingHarness(plant_canary(tmp_path))
    backend = FakeBackend(script)
    concierge = Concierge(
        store=store if store is not None else FlashStore(tmp_path / "flash" / "entries.jsonl"),
        researcher=researcher if researcher is not None else FakeResearcher(),
        harness=owned,
        approvals=approvals if approvals is not None else ApprovalGate(),
        backend_factory=lambda **_ignored: backend,
        output_fn=out.append,
        animated=False,
        input_fn=lambda _prompt: "",
        session_store=None,
        usage_preset=UsageLimits(max_wall_s=120, max_steps=40, max_model_calls=40),
    )
    return concierge, out, owned


def joined(lines: list[str]) -> str:
    return "\n".join(lines)


def visible(concierge: Concierge, lines: list[str]) -> str:
    """Operator text plus stored message bodies."""
    parts = list(lines)
    parts.extend(str(message.get("content", "")) for message in concierge.messages())
    return "\n".join(parts)


def assert_canary(path: Path) -> None:
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == CANARY_TEXT
