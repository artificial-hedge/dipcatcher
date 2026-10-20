"""``/help`` matches the commands the concierge actually dispatches."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fx1.interactive.concierge import HELP_TEXT, Concierge
from fx1.interactive.console import _SLASH_COMMANDS
from tests.fx1.ux_pty import hermetic_network

_COMMAND_RE = re.compile(r"^(\/[a-z][a-z0-9-]*)\b")


def _help_commands() -> list[str]:
    found: list[str] = []
    for line in HELP_TEXT.splitlines():
        match = _COMMAND_RE.match(line.strip())
        if match is None:
            continue
        name = match.group(1)
        if name not in found:
            found.append(name)
    return found


def _implemented() -> set[str]:
    names = set()
    for name in dir(Concierge):
        if name.startswith("_cmd_"):
            names.add("/" + name.removeprefix("_cmd_").replace("_", "-"))
    return names


def test_ux_help_lists_each_implemented_command(tmp_path: Path) -> None:
    del tmp_path
    listed = _help_commands()
    raw = _raw_commands()
    repeats = {name for name in raw if raw.count(name) > 1}
    assert repeats <= {"/superpower"}
    assert set(listed) == _implemented()
    assert set(_SLASH_COMMANDS) == set(listed)


def _raw_commands() -> list[str]:
    found: list[str] = []
    for line in HELP_TEXT.splitlines():
        match = _COMMAND_RE.match(line.strip())
        if match is not None:
            found.append(match.group(1))
    return found


def test_ux_help_commands_dispatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key, value in hermetic_network().items():
        monkeypatch.setenv(key, value)
    (tmp_path / "cfg").mkdir()
    (tmp_path / "sessions").mkdir()
    (tmp_path / "home").mkdir()
    said: list[str] = []
    concierge = Concierge(
        output_fn=lambda text="": said.append(str(text)),
        session_store=None,
        animated=False,
    )
    for command in _help_commands():
        said.clear()
        concierge.handle_line(command if command != "/exit" else "/help")
        err = capsys.readouterr().err
        assert "unknown command" not in err, command
        assert "unknown command" not in "\n".join(said), command
    assert concierge.handle_line("/exit") is True


def test_ux_unknown_command_does_not_invent_a_suggestion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    concierge = Concierge(output_fn=lambda text="": None, session_store=None, animated=False)
    concierge.handle_line("/superpwer")
    err = capsys.readouterr().err
    assert "unknown command /superpwer; did you mean /superpower? try /help" in err
