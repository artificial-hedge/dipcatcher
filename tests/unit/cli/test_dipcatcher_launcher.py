"""Public launcher contracts: chat in a terminal, unchanged automation otherwise."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dipcatcher_cli.app import main

ROOT = Path(__file__).resolve().parents[3]


def _terminal(monkeypatch: pytest.MonkeyPatch, *, stdin: bool, stdout: bool) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout)


@pytest.mark.parametrize(
    "args,model", [([], "fx1"), (["chat"], "fx1"), (["chat", "--model", "fx1-lite"], "fx1-lite")]
)
def test_terminal_routes_to_conversation(
    monkeypatch: pytest.MonkeyPatch, args: list[str], model: str
) -> None:
    from fx1.interactive import concierge

    seen: list[str] = []
    monkeypatch.setattr(concierge, "run_console", lambda *, model: seen.append(model))
    _terminal(monkeypatch, stdin=True, stdout=True)
    main(args)
    assert seen == [model]


@pytest.mark.parametrize("stdin,stdout", [(False, False), (True, False), (False, True)])
def test_bare_redirected_invocation_prints_help_without_chat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stdin: bool, stdout: bool
) -> None:
    _terminal(monkeypatch, stdin=stdin, stdout=stdout)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "Start a conversation: dipcatcher" in output
    assert "Usage:" in output
    assert "research" in output


def test_explicit_chat_refuses_redirected_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _terminal(monkeypatch, stdin=False, stdout=False)
    with pytest.raises(SystemExit) as stopped:
        main(["chat"])
    assert stopped.value.code == 2
    assert "requires an interactive terminal" in capsys.readouterr().err


def test_chat_help_is_available_without_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _terminal(monkeypatch, stdin=False, stdout=False)
    with pytest.raises(SystemExit) as stopped:
        main(["chat", "--help"])
    assert stopped.value.code == 0
    assert "fx1-lite" in capsys.readouterr().out


def test_invalid_model_is_rejected_before_chat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _terminal(monkeypatch, stdin=True, stdout=True)
    with pytest.raises(SystemExit) as stopped:
        main(["chat", "--model", "unknown"])
    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["research", "--help"], ["train"], ["does-not-exist"]])
def test_lab_commands_preserve_output_and_exit_code(args: list[str], tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NO_COLOR": "1", "TERM": "dumb"}
    env["FX1_CONFIG_DIR"] = str(tmp_path / "unused-config")
    legacy = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quant_fund.cli.main import app; app(prog_name='dipcatcher')",
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    launched = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (launched.returncode, launched.stdout, launched.stderr) == (
        legacy.returncode,
        legacy.stdout,
        legacy.stderr,
    )
    assert not Path(env["FX1_CONFIG_DIR"]).exists()


def test_help_does_not_import_interactive_model_code_or_read_stdin(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NO_COLOR": "1"}
    env["FX1_CONFIG_DIR"] = str(tmp_path / "unused-config")
    script = """
import sys
from dipcatcher_cli.app import main
try:
    main([])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name.startswith('fx1.interactive') for name in sys.modules)
assert sys.stdin.read() == '/superpower must-not-execute\\n'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input="/superpower must-not-execute\n",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["FX1_CONFIG_DIR"]).exists()
