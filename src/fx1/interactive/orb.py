"""Terminal port of the *thinking-orbs* dot choreography.

Inspired by rareformlabs' `thinking-orbs` (MIT, Jakub Antalik) — dotted
thought-orb indicators with six hand-tuned states — re-rendered for the
terminal as Braille pixels (2x4 dots per cell) in a single monochrome ink,
matching the original's strictly-monochrome discipline.

States and captions follow the original verbs. The animator is a no-op when
stdout is not a TTY, when ``NO_COLOR`` is set, when ``FXI_ORBS=off``, when
``TERM`` is ``dumb``, or when ``DIPCATCHER_PLAIN`` is set — pipes, tests,
and reduced-motion users get clean, static output.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import threading
import time
from enum import StrEnum
from typing import TextIO

from fx1.interactive.terminal import plain_requested, stream_lock

# Braille dot bit for (column, row) inside a 2x4 cell — U+2800 base.
_BRAILLE_BITS: dict[tuple[int, int], int] = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (0, 3): 0x40,
    (1, 3): 0x80,
}

ESC = "\x1b["
_DECSC = "\x1b7"
_DECRC = "\x1b8"
_ERASE_LINE = "\x1b[2K"

INK_LIGHT = "97"  # bright white ink on a dark background
INK_DARK = "30"  # black ink on a light background

DOTS = 12


class OrbState(StrEnum):
    """The six thinking-orbs verbs."""

    WORKING = "working"
    SEARCHING = "searching"
    SOLVING = "solving"
    LISTENING = "listening"
    COMPOSING = "composing"
    SHAPING = "shaping"


_CAPTIONS: dict[OrbState, str] = {
    OrbState.WORKING: "Working…",
    OrbState.SEARCHING: "Searching…",
    OrbState.SOLVING: "Solving…",
    OrbState.LISTENING: "Agent listening…",
    OrbState.COMPOSING: "Composing…",
    OrbState.SHAPING: "Agent shaping…",
}

_PRESETS: dict[str, tuple[int, int]] = {
    "avatar": (14, 4),  # chat-avatar scale
    "inline": (8, 2),  # inline-text scale
}


def caption(state: OrbState) -> str:
    return _CAPTIONS[state]


def orbs_enabled(stream: TextIO = sys.stdout) -> bool:
    if plain_requested():
        return False
    if os.environ.get("FXI_ORBS", "").strip().lower() in {"0", "off", "false", "no"}:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    is_tty = getattr(stream, "isatty", None)
    return bool(is_tty and is_tty())


def _orb_motion_allowed() -> bool:
    """Cursor motion is unsafe on a dumb terminal and in plain mode."""
    if plain_requested():
        return False
    return os.environ.get("TERM") != "dumb"


def _background_is_dark() -> bool:
    """Best-effort terminal background detection via COLORFGBG (fg;bg)."""
    raw = os.environ.get("COLORFGBG", "")
    if ";" in raw:
        try:
            bg = int(raw.rsplit(";", 1)[1])
        except ValueError:
            return True
        # 0-6 dark, 7/15 light; 8-14 are bright variants of the dark hues.
        return bg in {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14}
    return True  # terminals overwhelmingly default to dark


def _ink() -> str:
    return INK_LIGHT if _background_is_dark() else INK_DARK


def _dot_positions(state: OrbState, t: float, n: int = DOTS) -> list[tuple[float, float, float]]:
    """Dot centres as (angle, radius-fraction, intensity) at time *t*.

    Each state is a distinct choreography, faithful in spirit to the
    original: listening breathes, thinking chases, working pulses in a
    wave, searching sweeps like radar, solving converges and releases,
    composing runs two counter-rotating rings, shaping wobbles.
    """
    import math

    tau = math.tau
    base = [(tau * i / n, 1.0) for i in range(n)]
    out: list[tuple[float, float, float]] = []
    if state is OrbState.LISTENING:
        r = 1.0 + 0.07 * math.sin(t * 1.3)
        out = [
            (a, r, 0.75 + 0.25 * math.sin(t * 1.3 - tau * i / n)) for i, (a, _) in enumerate(base)
        ]
    elif state is OrbState.WORKING:
        out = [
            (a, 1.0, 0.35 + 0.65 * max(0.0, math.sin(t * 3.0 - tau * i / n)) ** 2)
            for i, (a, _) in enumerate(base)
        ]
    elif state is OrbState.SEARCHING:
        sweep = 2.0 * t
        out = [
            (a, 1.0, 0.25 + 0.75 * max(0.0, math.cos(sweep - a)) ** 3)
            for i, (a, _) in enumerate(base)
        ]
    elif state is OrbState.SOLVING:
        phase = 0.5 + 0.5 * math.sin(t * 1.1)
        ease = phase * phase * (3 - 2 * phase)
        out = [
            (a, 0.35 + 0.65 * min(1.0, ease + 0.25 * math.sin(t + i)), 0.8)
            for i, (a, _) in enumerate(base)
        ]
    elif state is OrbState.COMPOSING:
        half = n // 2
        out = [
            (tau * i / half + 0.5 * t, 1.0, 0.9)
            if i < half
            else (tau * (i - half) / (n - half) - 0.5 * t, 0.55, 0.9)
            for i in range(n)
        ]
    elif state is OrbState.SHAPING:
        out = [(a, 1.0 + 0.18 * math.sin(3 * a + 2.0 * t), 0.8) for a, _ in base]
    return out


def render_frame(
    state: OrbState,
    t: float | None = None,
    *,
    preset: str = "avatar",
    color: bool = True,
) -> str:
    """One static frame of the orb as a multi-line string (Braille pixels)."""
    import math

    if t is None:
        t = time.monotonic()
    width_chars, height_chars = _PRESETS[preset]
    width_px, height_px = width_chars * 2, height_chars * 4
    cx, cy = (width_px - 1) / 2, (height_px - 1) / 2
    radius = max(2.0, min(width_px, height_px) / 2 - 1.5)

    dots = _dot_positions(state, t)
    # Pixel coverage: a dot paints a filled circle whose radius scales with
    # its intensity — the original's only signal is dot size/placement.
    circles = [
        (cx + radius * r * math.cos(a), cy + radius * r * math.sin(a), 0.9 + 1.3 * s)
        for a, r, s in dots
    ]

    lines: list[str] = []
    for cy_char in range(height_chars):
        row = ""
        for cx_char in range(width_chars):
            bits = 0
            for dy in range(4):
                for dx in range(2):
                    px, py = cx_char * 2 + dx, cy_char * 4 + dy
                    covered = any((px - x) ** 2 + (py - y) ** 2 <= rad**2 for x, y, rad in circles)
                    if covered:
                        bits |= _BRAILLE_BITS[(dx, dy)]
            row += chr(0x2800 + bits) if bits else " "
        lines.append(row.rstrip())
    frame = "\n".join(lines)
    if color:
        return f"{ESC}{_ink()}m{frame}{ESC}0m"
    return frame


class OrbAnimator:
    """Animate an orb on its own line block while work is in flight.

    No-op unless ``orbs_enabled()``: non-TTY streams, ``NO_COLOR``, and
    ``FXI_ORBS=off`` all degrade to nothing. Frames share one clock via
    ``time.monotonic``, so concurrent orbs stay in phase.
    """

    def __init__(
        self,
        state: OrbState,
        *,
        preset: str = "avatar",
        fps: int = 12,
        stream: TextIO = sys.stdout,
    ) -> None:
        self.state = state
        self.preset = preset
        self.fps = fps
        self.stream = stream
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._painted = threading.Event()

    def __enter__(self) -> OrbAnimator:
        self._stop.clear()
        self._painted.clear()
        if orbs_enabled(self.stream) and _orb_motion_allowed():
            self._thread = threading.Thread(target=self._loop, name="fx1-orb", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=1.0)
        if thread.is_alive() or not self._painted.is_set():
            return
        self._clear_block()

    def _loop(self) -> None:
        interval = 0.05 if self.fps <= 0 else 1.0 / self.fps
        lock = stream_lock(self.stream)
        first = True
        while not self._stop.is_set():
            frame = render_frame(self.state, preset=self.preset)
            with lock:
                if self._stop.is_set():
                    return
                self._write_frame(frame, first=first)
                first = False
                self._painted.set()
            if self._stop.wait(interval):
                return

    def _write_frame(self, frame: str, *, first: bool) -> None:
        self.stream.write(_DECSC if first else _DECRC)
        self.stream.write(frame)
        self.stream.write("\n")
        self.stream.flush()

    def _clear_block(self) -> None:
        height = _PRESETS.get(self.preset, (8, 2))[1]
        parts = [_DECRC]
        for index in range(height):
            parts.append(_ERASE_LINE)
            if index + 1 < height:
                parts.append(f"{ESC}1B")
        parts.append(_DECRC)
        with contextlib.suppress(OSError), stream_lock(self.stream):
            self.stream.write("".join(parts))
            self.stream.flush()


def banner(model: str, host: str, *, color: bool = True) -> str:
    """Static startup banner: a listening orb beside the harness title."""
    if not _orb_motion_allowed():
        color = False
    orb_lines = render_frame(OrbState.LISTENING, preset="inline", color=False).split("\n")
    orb_w = max((len(line) for line in orb_lines), default=0)
    title = f"dipcatcher · fx-1 interactive — model {model} @ {host}"
    text_w, inner = _banner_widths(title, orb_w)
    rows = [_banner_edge("╭", "╮", inner)]
    rows.extend(_banner_rows(orb_lines, orb_w, title, text_w, color=color))
    rows.append(_banner_edge("╰", "╯", inner))
    return "\n".join(rows)


def _banner_widths(title: str, orb_w: int) -> tuple[int, int]:
    columns = max(20, shutil.get_terminal_size((80, 24)).columns)
    text_w = len(title)
    room = columns - 2 - orb_w - 2
    if room > text_w:
        text_w = room
    return text_w, orb_w + 2 + text_w


def _banner_edge(left: str, right: str, inner: int) -> str:
    return left + ("─" * inner) + right


def _banner_rows(
    lines: list[str],
    orb_w: int,
    title: str,
    text_w: int,
    *,
    color: bool,
) -> list[str]:
    ink = _ink()
    rows: list[str] = []
    for index, line in enumerate(lines):
        cell = line.ljust(orb_w)
        if color:
            cell = f"{ESC}{ink}m{cell}{ESC}0m"
        label = title if index == 0 else ""
        rows.append(f"│{cell}  {label.ljust(text_w)}│")
    return rows
