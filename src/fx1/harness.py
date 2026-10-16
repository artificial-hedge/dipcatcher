"""dipcatcher as the fx-1 harness.

The relationship, inverted: **fx-1 is the model; dipcatcher is the harness
that builds, evaluates, and verifies it.** This module is the typed bridge —
every lab capability fx-1 is allowed to touch is registered here as a
:class:`HarnessCommand` and executed through an injectable runner, so tests
never spawn real lab processes and fx-1 can never reach an unregistered
command.

Harness roles:
- *Data engine* — ingest/collect/features/labels feed ``fx1.data``.
- *Evaluation* — benches and books score fx-1's domain competence;
  ``fx1.eval`` scores its behavior.
- *Verification* — doctor / verify-research / validate / monitor gate every
  artifact the model is trained on or claims about.
- *Model training* — the lab's own train/optimize/paper surfaces, usable as
  curriculum and tool-use trace sources for fx-1.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # noqa: S404 - runner is injectable; see Harness.run
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO

from pydantic import BaseModel, Field


class HarnessRole(StrEnum):
    DATA_ENGINE = "data_engine"
    EVALUATION = "evaluation"
    VERIFICATION = "verification"
    MODEL_TRAINING = "model_training"


class HarnessCommand(BaseModel):
    """A registered lab command fx-1 may invoke through the harness."""

    name: str
    role: HarnessRole
    argv: list[str] = Field(description="dipcatcher CLI argv (no executable)")
    timeout_s: int = Field(default=300, ge=1, le=7200)
    description: str


def _cmd(name: str, role: HarnessRole, description: str, timeout_s: int = 300) -> HarnessCommand:
    return HarnessCommand(
        name=name, role=role, argv=[name], timeout_s=timeout_s, description=description
    )


# The complete registry of lab surfaces fx-1 can touch — mirrors
# ``quant_fund.cli.main``. Anything not listed here is unreachable by the
# model: fail-closed by construction. The hidden ``lab`` command and the
# network-serving ``api`` command are deliberately excluded.
HARNESS_REGISTRY: tuple[HarnessCommand, ...] = (
    # --- Verification ---
    _cmd(
        "doctor",
        HarnessRole.VERIFICATION,
        "Verify data manifest and latest research receipt before trusting state.",
    ),
    _cmd(
        "verify-research",
        HarnessRole.VERIFICATION,
        "Verify immutable provenance, scorecards, and research artifacts.",
    ),
    _cmd(
        "validate",
        HarnessRole.VERIFICATION,
        "Fail-closed causal / walk-forward / promotion gates for a model id.",
    ),
    _cmd(
        "monitor",
        HarnessRole.VERIFICATION,
        "Operational monitoring surface for the harness runtime.",
    ),
    _cmd(
        "research-families",
        HarnessRole.VERIFICATION,
        "Read-only benchmark family catalog (required / live optional / retired) as JSON.",
        timeout_s=120,
    ),
    # --- Data engine ---
    _cmd(
        "ingest",
        HarnessRole.DATA_ENGINE,
        "Bronze/silver lake ingest; writes data_manifest.json with SHA-256 hashes.",
        timeout_s=1800,
    ),
    _cmd(
        "collect",
        HarnessRole.DATA_ENGINE,
        "Explicit opt-in public-feed collection (binance, fred) with receipts.",
    ),
    _cmd(
        "build-features",
        HarnessRole.DATA_ENGINE,
        "Feature construction on the PIT lake.",
        timeout_s=1800,
    ),
    _cmd(
        "build-labels",
        HarnessRole.DATA_ENGINE,
        "Label construction on the PIT lake.",
        timeout_s=1800,
    ),
    # --- Evaluation ---
    _cmd(
        "research",
        HarnessRole.EVALUATION,
        "Scientific benches (proper scores; SYNTHETIC labeled).",
        timeout_s=3600,
    ),
    _cmd(
        "northset",
        HarnessRole.EVALUATION,
        "Order-book/candlestick slice: identities, OHLC vol, Kyle/Roll/OFI/VPIN.",
        timeout_s=1800,
    ),
    _cmd(
        "backtest",
        HarnessRole.EVALUATION,
        "Next-open fills, costs, participation; tamper-evident artifacts.",
        timeout_s=1800,
    ),
    _cmd("forecast", HarnessRole.EVALUATION, "Forecast surface for validated models."),
    _cmd(
        "kronos-forecast",
        HarnessRole.EVALUATION,
        "Research-only local Kronos candlestick adapter (no network).",
    ),
    _cmd(
        "candle-book",
        HarnessRole.EVALUATION,
        "Candle/order-book reconstruction benches.",
        timeout_s=900,
    ),
    _cmd(
        "kyle-ofi",
        HarnessRole.EVALUATION,
        "Kyle lambda / OFI microstructure benches.",
        timeout_s=900,
    ),
    _cmd(
        "session-book", HarnessRole.EVALUATION, "Session-level order-book benches.", timeout_s=900
    ),
    _cmd("vendor-book-map", HarnessRole.EVALUATION, "Vendor book mapping diagnostics."),
    _cmd("book-panel", HarnessRole.EVALUATION, "Book panel aggregation benches.", timeout_s=900),
    _cmd("report", HarnessRole.EVALUATION, "Research notebook/report generation."),
    _cmd(
        "tearsheet",
        HarnessRole.EVALUATION,
        "Backtest tearsheet diagnostics (analytics export, live_pnl_claim=false).",
    ),
    # --- Model training (lab surfaces; curriculum + tool-use traces) ---
    _cmd(
        "train",
        HarnessRole.MODEL_TRAINING,
        "Lab model training families (ranking/distribution/calibration/"
        "volatility/alpha/covariance/regime/tail/reinforcement/liquidity).",
        timeout_s=7200,
    ),
    _cmd(
        "optimize",
        HarnessRole.MODEL_TRAINING,
        "Hyperparameter optimization under fail-closed gates.",
        timeout_s=7200,
    ),
    _cmd(
        "paper",
        HarnessRole.MODEL_TRAINING,
        "Simulated broker shadow runs (no live fills).",
        timeout_s=1800,
    ),
)


class HarnessResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    cancelled: bool = False
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.cancelled and not self.timed_out


# Runner signature: (argv, timeout_s) -> (exit_code, stdout, stderr)
Runner = Callable[[list[str], int], tuple[int, str, str]]

_MISSING_EXECUTABLE = (
    "dipcatcher executable not found on PATH; install the fx-1 distribution "
    "(uv sync) or inject a runner"
)
_READ_CHUNK = 8192


@dataclass(frozen=True)
class RunnerOutcome:
    exit_code: int
    stdout: str
    stderr: str
    cancelled: bool = False
    timed_out: bool = False


CancellableRunner = Callable[[list[str], int, threading.Event | None], RunnerOutcome]


class _OutputBuffer:
    """Bounded text capture; excess characters are discarded while the pipe is drained."""

    def __init__(self, limit: int) -> None:
        self._limit = limit if limit > 0 else 0
        self._parts: list[str] = []
        self._size = 0
        self._truncated = False

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        if self._size >= self._limit:
            self._truncated = True
            return
        room = self._limit - self._size
        if len(chunk) <= room:
            self._parts.append(chunk)
            self._size += len(chunk)
            return
        if room > 0:
            self._parts.append(chunk[:room])
            self._size += room
        self._truncated = True

    def text(self) -> str:
        body = "".join(self._parts)
        if not self._truncated:
            return body
        return f"{body}\n[output truncated at {self._limit} chars]"


def default_runner_available() -> bool:
    """True when the ``dipcatcher`` console script is on ``PATH``."""
    return shutil.which("dipcatcher") is not None


def _adapt_legacy_runner(runner: Runner) -> CancellableRunner:
    def _run(
        argv: list[str],
        timeout_s: int,
        _cancel_event: threading.Event | None,
    ) -> RunnerOutcome:
        code, out, err = runner(argv, timeout_s)
        return RunnerOutcome(exit_code=code, stdout=out, stderr=err)

    return _run


def _spawn(executable: str, argv: list[str]) -> subprocess.Popen[str]:
    command = [executable, *argv]
    if os.name == "posix":
        return subprocess.Popen(  # noqa: S603 - executable is injected or the dipcatcher CLI
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            start_new_session=True,
        )
    return subprocess.Popen(  # noqa: S603 - executable is injected or the dipcatcher CLI
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _drain(stream: IO[str], sink: _OutputBuffer) -> None:
    try:
        while True:
            chunk = stream.read(_READ_CHUNK)
            if chunk == "":
                return
            sink.append(chunk)
    except (OSError, ValueError):
        return


def _signal_direct(proc: subprocess.Popen[str], *, kill: bool) -> None:
    try:
        if kill:
            proc.kill()
        else:
            proc.terminate()
    except OSError:
        return


def _signal_posix(proc: subprocess.Popen[str], *, kill: bool) -> None:
    # Windows stubs omit SIGKILL, killpg, and getpgid. Callers skip this
    # function when os.name is not posix.
    sig = getattr(signal, "SIGKILL", signal.SIGTERM) if kill else signal.SIGTERM
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is None or getpgid is None:
        _signal_direct(proc, kill=kill)
        return
    try:
        killpg(getpgid(proc.pid), sig)
    except OSError:
        _signal_direct(proc, kill=kill)


def _signal_process(proc: subprocess.Popen[str], *, kill: bool) -> None:
    if os.name == "posix":
        _signal_posix(proc, kill=kill)
        return
    _signal_direct(proc, kill=kill)


def _wait_for_exit(proc: subprocess.Popen[str], grace_s: float) -> None:
    if proc.poll() is not None:
        return
    timeout = grace_s if grace_s > 0 else 0.0
    try:
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return


def _escalate(proc: subprocess.Popen[str], grace_s: float) -> None:
    """SIGTERM the child, then SIGKILL after ``grace_s``. Never raises if it is already gone."""
    try:
        if proc.poll() is not None:
            return
        _signal_process(proc, kill=False)
        _wait_for_exit(proc, grace_s)
        if proc.poll() is not None:
            return
        _signal_process(proc, kill=True)
        _wait_for_exit(proc, grace_s if grace_s > 0 else 1.0)
    except OSError:
        _signal_direct(proc, kill=True)


def _pause(
    cancel_event: threading.Event | None,
    poll_s: float,
    remaining_s: float,
) -> None:
    delay = poll_s if remaining_s > poll_s else remaining_s
    if delay <= 0:
        time.sleep(0)
        return
    if cancel_event is None:
        time.sleep(delay)
        return
    cancel_event.wait(delay)


def _close_streams(proc: subprocess.Popen[str]) -> None:
    for stream in (proc.stdout, proc.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            continue


def _join_readers(threads: list[threading.Thread], proc: subprocess.Popen[str]) -> None:
    for thread in threads:
        thread.join(timeout=2.0)
    if not any(thread.is_alive() for thread in threads):
        _close_streams(proc)
        return
    _close_streams(proc)
    for thread in threads:
        thread.join(timeout=1.0)


def _start_readers(
    proc: subprocess.Popen[str],
    stdout_buf: _OutputBuffer,
    stderr_buf: _OutputBuffer,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    if proc.stdout is not None:
        threads.append(
            threading.Thread(
                target=_drain,
                args=(proc.stdout, stdout_buf),
                name="harness-stdout",
                daemon=True,
            )
        )
    if proc.stderr is not None:
        threads.append(
            threading.Thread(
                target=_drain,
                args=(proc.stderr, stderr_buf),
                name="harness-stderr",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    return threads


def _wait_for_child(
    proc: subprocess.Popen[str],
    *,
    timeout_s: int,
    cancel_event: threading.Event | None,
    poll_s: float,
    grace_s: float,
    started: float,
) -> tuple[bool, bool]:
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            _escalate(proc, grace_s)
            return True, False
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            _escalate(proc, grace_s)
            return False, True
        _pause(cancel_event, poll_s, float(timeout_s) - elapsed)
    return False, False


def _run_subprocess(
    executable: str,
    argv: list[str],
    timeout_s: int,
    cancel_event: threading.Event | None,
    *,
    grace_s: float,
    poll_s: float,
    max_output_chars: int,
) -> RunnerOutcome:
    proc = _spawn(executable, argv)
    stdout_buf = _OutputBuffer(max_output_chars)
    stderr_buf = _OutputBuffer(max_output_chars)
    started = time.monotonic()
    cancelled = False
    timed_out = False
    threads: list[threading.Thread] = []
    try:
        threads = _start_readers(proc, stdout_buf, stderr_buf)
        cancelled, timed_out = _wait_for_child(
            proc,
            timeout_s=timeout_s,
            cancel_event=cancel_event,
            poll_s=poll_s,
            grace_s=grace_s,
            started=started,
        )
    finally:
        if proc.poll() is None:
            _escalate(proc, grace_s)
        _join_readers(threads, proc)
    code = proc.returncode if proc.returncode is not None else proc.poll()
    if code is None:
        code = 1
    return RunnerOutcome(
        exit_code=code,
        stdout=stdout_buf.text(),
        stderr=stderr_buf.text(),
        cancelled=cancelled,
        timed_out=timed_out,
    )


def make_subprocess_runner(
    executable: str = "dipcatcher",
    *,
    grace_s: float = 5.0,
    poll_s: float = 0.05,
    max_output_chars: int = 2_000_000,
) -> CancellableRunner:
    """Subprocess runner that polls for cancel and enforces ``timeout_s``.

    POSIX children are started in a new session so the whole process group can
    be signalled. Windows children are stopped with ``terminate`` / ``kill``.
    """

    def _run(
        argv: list[str],
        timeout_s: int,
        cancel_event: threading.Event | None,
    ) -> RunnerOutcome:
        return _run_subprocess(
            executable,
            argv,
            timeout_s,
            cancel_event,
            grace_s=grace_s,
            poll_s=poll_s,
            max_output_chars=max_output_chars,
        )

    return _run


def _subprocess_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    outcome = make_subprocess_runner()(argv, timeout_s, None)
    return outcome.exit_code, outcome.stdout, outcome.stderr


def _resolve_timeout(default_s: int, timeout_s: int | None) -> int:
    if timeout_s is None:
        return default_s
    if timeout_s < 1:
        raise ValueError(f"timeout_s must be >= 1, got {timeout_s}")
    return min(default_s, timeout_s)


class Harness:
    """The fx-1 side of the dipcatcher harness."""

    def __init__(
        self,
        runner: Runner | None = None,
        *,
        cancellable_runner: CancellableRunner | None = None,
    ) -> None:
        self._use_default_runner = runner is None and cancellable_runner is None
        chosen: CancellableRunner
        if runner is not None:
            chosen = _adapt_legacy_runner(runner)
        elif cancellable_runner is not None:
            chosen = cancellable_runner
        else:
            chosen = make_subprocess_runner()
        self._runner = chosen
        self._registry = {c.name: c for c in HARNESS_REGISTRY}

    def list_commands(self, role: HarnessRole | None = None) -> list[HarnessCommand]:
        commands = list(self._registry.values())
        if role is not None:
            commands = [c for c in commands if c.role == role]
        return commands

    def get(self, name: str) -> HarnessCommand:
        if name not in self._registry:
            raise KeyError(
                f"{name!r} is not a registered harness command; fx-1 may only "
                f"use: {sorted(self._registry)}"
            )
        return self._registry[name]

    def run(
        self,
        name: str,
        extra_args: list[str] | None = None,
        *,
        config: Path | None = None,
        cancel_event: threading.Event | None = None,
        timeout_s: int | None = None,
    ) -> HarnessResult:
        """Execute a registered harness command, fail-closed on unknown names.

        Config paths are contained to the repo ``configs/`` directory, matching
        the lab API's allowlist posture. ``timeout_s`` may only tighten the
        registry timeout. The default subprocess runner refuses to spawn when
        ``dipcatcher`` is not on ``PATH``.
        """
        command = self.get(name)
        effective_timeout = _resolve_timeout(command.timeout_s, timeout_s)
        argv = list(command.argv)
        configs_dir = (Path.cwd() / "configs").resolve()

        def check_contained(raw: str) -> Path:
            resolved = Path(raw).resolve()
            if configs_dir not in resolved.parents and resolved != configs_dir:
                raise ValueError(f"config path {resolved} escapes the configs/ allowlist")
            return resolved

        if config is not None:
            argv += ["--config", str(check_contained(str(config)))]
        if extra_args:
            # --config passed through extra_args must obey the same
            # containment as the keyword arg — otherwise it is a bypass.
            for i, arg in enumerate(extra_args):
                if arg == "--config" and i + 1 < len(extra_args):
                    check_contained(extra_args[i + 1])
                elif arg.startswith("--config="):
                    check_contained(arg.split("=", 1)[1])
            argv += list(extra_args)
        if self._use_default_runner and not default_runner_available():
            raise RuntimeError(_MISSING_EXECUTABLE)
        started = time.monotonic()
        outcome = self._runner(argv, effective_timeout, cancel_event)
        duration_s = time.monotonic() - started
        return HarnessResult(
            command=name,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            cancelled=outcome.cancelled,
            timed_out=outcome.timed_out,
            duration_s=duration_s,
        )
