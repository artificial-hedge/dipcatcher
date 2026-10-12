"""Real ``python -m dipcatcher_cli chat`` under awkward terminals."""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
from pathlib import Path

import pytest

from tests.fx1.ux_pty import prepare, run_pty

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX pty")

_PROMPT = "fx1 › ".encode()
_ASCII_PROMPT = b"fx1 > "
_PASTE_OFF = b"\x1b[?2004l"
_SECRET = "sk-history-must-not-leak"


def _exit(
    root_name: str,
    tmp_path: Path,
    actions: list[tuple[object, ...]],
    *,
    paste: bool = True,
    **kwargs: object,
):
    root = prepare(tmp_path, root_name)
    result = run_pty(root, actions, **kwargs)  # type: ignore[arg-type]
    assert "Traceback" not in result.text
    assert "TERMIOS_OK" in result.text
    if paste:
        assert _PASTE_OFF in result.raw
    return result


def test_ux_banner_prompt_and_termios(tmp_path: Path) -> None:
    result = _exit(
        "banner",
        tmp_path,
        [("wait", _PROMPT), ("send", b"/exit\n"), ("wait", b"TERMIOS_OK")],
    )
    assert result.status == 0
    assert "dipcatcher" in result.text
    assert "/help, /superpower, /exit" in result.text


@pytest.mark.parametrize("cols", [20, 40, 80, 200])
def test_ux_narrow_tables_do_not_crash(tmp_path: Path, cols: int) -> None:
    result = _exit(
        f"cols{cols}",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"/models\n"),
            ("wait", b"\x1b[?2004h"),
            ("send", b"/sessions\n"),
            ("wait", b"\x1b[?2004h"),
            ("send", b"/evidence\n"),
            ("wait", b"\x1b[?2004h"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
        cols=cols,
        rows=5,
    )
    assert result.status == 0
    assert "ValueError" not in result.text


@pytest.mark.parametrize(
    ("name", "extra", "prompt", "sgr", "paste"),
    [
        ("nocolor", {"NO_COLOR": "1", "FORCE_COLOR": "1"}, _PROMPT, False, True),
        ("dumb", {"TERM": "dumb"}, "dip › ".encode(), False, False),
        ("force0", {"FORCE_COLOR": "0"}, _PROMPT, True, True),
        ("unset-term", {"TERM": None}, _PROMPT, True, True),
    ],
)
def test_ux_color_switches(
    tmp_path: Path,
    name: str,
    extra: dict[str, str | None],
    prompt: bytes,
    sgr: bool,
    paste: bool,
) -> None:
    result = _exit(
        name,
        tmp_path,
        [("wait", prompt), ("send", b"/exit\n"), ("wait", b"TERMIOS_OK")],
        extra=extra,
        paste=paste,
    )
    assert result.status == 0
    plain = result.raw.replace(b"\x1b[?2004h", b"").replace(b"\x1b[?2004l", b"")
    if sgr:
        assert b"\x1b[" in plain
    else:
        assert b"\x1b" not in plain
    assert b"\x1b7" not in result.raw


@pytest.mark.parametrize(
    ("name", "extra", "ascii_chrome"),
    [
        ("lang-c", {"LANG": "C", "LC_ALL": "C", "TERM": "xterm"}, False),
        ("posix", {"LANG": "POSIX", "LC_ALL": "POSIX", "TERM": "xterm"}, False),
        ("ascii-io", {"PYTHONIOENCODING": "ascii", "LANG": "C", "LC_ALL": "C"}, True),
        ("latin1", {"PYTHONIOENCODING": "iso-8859-1", "LANG": "C", "LC_ALL": "C"}, True),
    ],
)
def test_ux_ascii_locale_keeps_the_prompt(
    tmp_path: Path, name: str, extra: dict[str, str], ascii_chrome: bool
) -> None:
    prompt = _ASCII_PROMPT if ascii_chrome else _PROMPT
    result = _exit(
        name,
        tmp_path,
        [
            ("wait", prompt),
            ("send", b"/help\n"),
            ("wait", b"Chat"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
        extra=extra,
    )
    assert result.status == 0
    assert "can't encode" not in result.text
    assert len(result.raw) < 100_000
    if ascii_chrome:
        assert "dipcatcher |" in result.text
        assert "╭" not in result.text


def test_ux_keys_set_never_hits_history(tmp_path: Path) -> None:
    root = prepare(tmp_path, "keys")
    result = run_pty(
        root,
        [
            ("wait", _PROMPT),
            ("send", b"recall-token\n"),
            ("wait", b"endpoint"),
            ("send", b"/keys set fx1\n"),
            ("wait", b"API key"),
            ("send", _SECRET.encode() + b"\n"),
            ("wait", b"endpoint URL"),
            ("send", b"http://127.0.0.1:9/v1\n"),
            ("wait", b"stored"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0
    assert "TERMIOS_OK" in result.text
    history = (root / "cfg" / "history").read_text(encoding="utf-8")
    assert "/keys" not in history
    assert _SECRET not in history
    assert "recall-token" in history
    mode = (root / "cfg" / "history").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_ux_editing_paste_and_wide_chars(tmp_path: Path) -> None:
    wide = "你a\u0301".encode() + b"\x7f\n"
    paste = b"\x1b[200~alpha\r\nbeta\x1b[201~\n"
    marked = b"\x1b[200~hello\x1b[201~tail\x1b[201~\n"
    result = _exit(
        "edit",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"/hel\t\n"),
            ("wait", b"plain text"),
            ("send", b"ab\x01X\n"),
            ("wait", b"Xab"),
            ("send", paste),
            ("wait", b"alpha"),
            ("send", marked),
            ("wait", b"hello"),
            ("send", wide),
            ("wait", "你".encode()),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0
    assert "beta" in result.text


def test_ux_hundred_kilobyte_paste(tmp_path: Path) -> None:
    blob = b"\x1b[200~" + (b"Z" * 100_000) + b"\x1b[201~\n"
    result = _exit(
        "paste100",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", blob),
            ("wait", b"Slash commands still work"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0
    assert result.raw.count(b"Z") >= 100_000


def test_ux_ctrl_d_empty_exits(tmp_path: Path) -> None:
    result = _exit(
        "ctld",
        tmp_path,
        [("wait", _PROMPT), ("send", b"\x04"), ("wait", b"TERMIOS_OK")],
    )
    assert result.status == 0


def test_ux_ctrl_d_on_text_does_not_exit(tmp_path: Path) -> None:
    result = _exit(
        "ctld2",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"abc\x04\n"),
            ("wait", b"endpoint"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0
    assert "abc" in result.text


def test_ux_double_ctrl_c(tmp_path: Path) -> None:
    result = _exit(
        "ctrlc",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"\x03"),
            ("wait", b"press Ctrl-C again"),
            ("send", b"\x03"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0


def test_ux_sigtstp_round_trip(tmp_path: Path) -> None:
    result = _exit(
        "tstp",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("signal", signal.SIGTSTP),
            ("signal", signal.SIGCONT),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0


def test_ux_sigquit_restores_the_tty(tmp_path: Path) -> None:
    result = _exit(
        "quit",
        tmp_path,
        [("wait", _PROMPT), ("signal", signal.SIGQUIT), ("wait", b"TERMIOS_OK")],
    )
    assert result.status == 128 + signal.SIGQUIT


def test_ux_sigterm_restores_the_tty(tmp_path: Path) -> None:
    result = _exit(
        "term",
        tmp_path,
        [("wait", _PROMPT), ("signal", signal.SIGTERM), ("wait", b"TERMIOS_OK")],
    )
    assert result.status == 143


def test_ux_sigwinch_keeps_the_prompt(tmp_path: Path) -> None:
    result = _exit(
        "winch",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("resize", 12, 40),
            ("send", b"/models\n"),
            ("wait", b"\x1b[?2004h"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
        cols=80,
        rows=24,
    )
    assert result.status == 0


def test_ux_job_status_keeps_typed_text(tmp_path: Path) -> None:
    result = _exit(
        "job",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"/research ux-background-job\n"),
            ("wait", b"running in the background"),
            ("send", b"QQ"),
            ("wait", b"QQ"),
            ("send", b"\x15/status\n"),
            ("wait", b"[research]"),
            ("send", b"/cancel\n"),
            ("wait", b"sources fetched"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0
    sent = next(stamp for stamp, _kind, data in result.events if data == b"\x15/status\n")
    seen = next(
        stamp for stamp, kind, data in result.events if kind == "wait" and data == b"[research]"
    )
    assert seen - sent < 10
    region = result.raw.split(b"running in the background", 1)[1]
    assert region.count(b"QQ") >= 1
    assert "[research]" in result.text


def test_ux_model_error_restores_termios(tmp_path: Path) -> None:
    result = _exit(
        "modelerr",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"hello\n"),
            ("wait", b"endpoint"),
            ("send", b"/exit\n"),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 0
    assert "Traceback" not in result.text


def _pids_gone(pids: list[int]) -> None:
    for pid in pids:
        assert not Path(f"/proc/{pid}").exists(), pid


def _sessions_not_running(root: Path) -> int:
    count = 0
    for path in (root / "sessions").glob("*.json"):
        count += 1
        job = json.loads(path.read_text(encoding="utf-8")).get("job") or {}
        assert job.get("state") != "running", path.read_text(encoding="utf-8")
    return count


def test_ux_sighup_at_the_prompt_exits_129(tmp_path: Path) -> None:
    result = _exit(
        "hup-prompt",
        tmp_path,
        [("wait", _PROMPT), ("signal", signal.SIGHUP), ("wait", b"TERMIOS_OK")],
    )
    assert result.status == 129
    assert "Traceback" not in result.text
    _pids_gone([result.child, *result.orphans])
    _sessions_not_running(result.root)


def test_ux_sighup_during_a_job_does_not_leave_it_running(tmp_path: Path) -> None:
    result = _exit(
        "hup-job",
        tmp_path,
        [
            ("wait", _PROMPT),
            ("send", b"/research ux-hup-job\n"),
            ("wait", b"running in the background"),
            ("signal", signal.SIGHUP),
            ("wait", b"TERMIOS_OK"),
        ],
    )
    assert result.status == 129
    assert "Traceback" not in result.text
    assert _sessions_not_running(result.root) >= 1
    _pids_gone([result.child, *result.orphans])


_HOSTILE = r"""
from fx1.interactive.concierge import Concierge
from fx1.interactive.console import run_console


def handle(self: Concierge, line: str) -> bool:
    text = line.strip()
    if text in {"/exit", "exit", "quit"}:
        self.shutdown()
        return True
    self._say("SAFE\x1b]0;pwn\x07\x1b[2J\x9b1Htail")
    return False


Concierge.handle_line = handle  # type: ignore[method-assign]

if __name__ == "__main__":
    run_console()
"""


def test_ux_say_strips_hostile_controls_on_a_pty(tmp_path: Path) -> None:
    root = prepare(tmp_path, "hostile")
    drive = root / "drive.py"
    drive.write_text(_HOSTILE, encoding="utf-8")
    result = run_pty(
        root,
        [
            ("wait", _PROMPT),
            ("send", b"x\n"),
            ("wait", b"SAFEtail"),
            ("send", b"/exit\n"),
        ],
        command=[sys.executable, str(drive)],
        wrap=False,
        extra={"NO_COLOR": "1"},
    )
    assert result.status == 0
    assert b"SAFEtail" in result.raw
    assert b"\x1b]0;" not in result.raw
    assert b"\x1b[2J" not in result.raw
    assert b"\x9b" not in result.raw
