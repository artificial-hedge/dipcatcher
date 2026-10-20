#!/usr/bin/env python3
"""Measure dipcatcher chat startup and local input latency.

Writes one JSON document (``dipcatcher.latency.v1``) with cold-prompt
samples, local slash-command acknowledgement samples, and cancel-to-stop
for an owned sleeper. Proposed P5 limits are scored on this machine.
``reference_hardware`` stays false unless ``--reference-hardware`` is
passed; passing the flag does not identify reference hardware and does
not close the P5 gate.

``python -X importtime`` runs only under ``--import-profile``. Timing
runs do not enable it.

Cancel-to-stop uses ``fx1.interactive.procs.run_owned`` on this script's
``--sleeper`` child. Chat ``/cancel`` is not timed: nothing in the
product starts a slow local job without a product change.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "dipcatcher.latency.v1"
PROMPT = "dip › "
_FALLBACK_MARKERS: tuple[str, ...] = (
    "fx1-lite › ",
    "fx1 › ",
    "fx1-lite > ",
    "fx1 > ",
    "dip > ",
    PROMPT,
)
PERCENTILE_METHOD = "nearest_rank"
DEFAULT_RUNS = 30
DEFAULT_EVENTS = 200
MAX_RUNS = 1000
MAX_EVENTS = 5000
MAX_TIMEOUT_S = 120.0
COLD_P95_LIMIT_S = 2.0
COLD_REQUIRED_SAMPLES = 30
INPUT_P95_LIMIT_S = 0.1
INPUT_REQUIRED_SAMPLES = 200
CANCEL_LIMIT_S = 1.0
CANCEL_REQUIRED_SAMPLES = 1
OWNED_STOP_LIMIT_S = 5.0
CANCEL_POLL_S = 0.05
TRANSIENT_RETRY_S = 30.0
IMPORT_TOP_N = 20
POSIX_PTY_ERROR = "PTY measurement requires POSIX"
CANCEL_METHOD = "fx1.interactive.procs.run_owned"
CANCEL_REASON = (
    "Chat /cancel was not timed. No product seam starts a slow local job "
    "without a product edit, so this is cancel-to-stop for a sleeper owned "
    "by fx1.interactive.procs.run_owned (default cancel_poll_s)."
)
NOTE = (
    "Proposed P5 targets from docs/DIPCATCHER_PRODUCT_CONTRACT.md, measured "
    "on this machine. reference_hardware is true only when "
    "--reference-hardware was passed; this tool does not identify reference "
    "hardware. p5_gate_closed stays false: crash/restart coverage and "
    "separate model-latency measurement are out of scope."
)
LOCAL_COMMANDS: tuple[str, ...] = ("/help", "/status", "/model", "/session")
_READ_SIZE = 16384
_TAIL_LIMIT = 4000
_FAILURE_LIMIT = 500
_IMPORT_CAP = 2_000_000
_DROP_EXACT = frozenset(
    {
        "moonshot_api_key",
        "moonshot_api_url",
        "fx1_base_url",
        "dipcatcher_fx1_api_key",
        "dipcatcher_fx1_base_url",
        "dipcatcher_fx1_lite_api_key",
        "dipcatcher_fx1_lite_base_url",
    }
)
_DROP_SUFFIXES = ("_api_key", "_token", "_secret", "_password")
_IMPORT_LINE = re.compile(r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s*(.*?)\s*$")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
_SRC = REPO_ROOT / "src"

SleepFn = Callable[[float], None]
Runner = Callable[..., Any]
ColdFn = Callable[["MeasureRequest", Mapping[str, str]], Mapping[str, Any]]
ImportFn = Callable[["MeasureRequest", Mapping[str, str]], str]
FactsFn = Callable[[], Mapping[str, Any]]


class UsageError(Exception):
    """CLI arguments do not match the measurement contract."""


class LatencyError(Exception):
    """A measurement could not be collected."""


@dataclass(frozen=True, slots=True)
class MeasureRequest:
    """One measurement invocation. Empty ``python`` means ``sys.executable``."""

    runs: int = DEFAULT_RUNS
    events: int = DEFAULT_EVENTS
    reference_hardware: bool = False
    cold_timeout_s: float = 30.0
    event_timeout_s: float = 5.0
    cancel_timeout_s: float = 10.0
    python: str = ""


@dataclass(frozen=True, slots=True)
class _Options:
    request: MeasureRequest
    out: Path | None
    import_profile: bool


@dataclass(slots=True)
class _Retry:
    used: bool = False


def pty_supported() -> bool:
    """Real pseudo-terminals are a POSIX facility."""
    return os.name == "posix"


class _ModelStub:
    """Enough of a concierge for ``console._engine_prompt``."""

    def __init__(self, model: str) -> None:
        self.model = model


def prompt_pattern_from(markers: Sequence[str]) -> str:
    """Alternation of prompt literals, longest first so ``fx1-lite`` stays intact."""
    ordered = sorted({item for item in markers if item}, key=lambda item: (-len(item), item))
    if not ordered:
        raise ValueError("prompt pattern needs at least one marker")
    return "(?:" + "|".join(re.escape(item) for item in ordered) + ")"


FALLBACK_PROMPT_PATTERN = prompt_pattern_from(_FALLBACK_MARKERS)
_ARMED: dict[str, Any] = {}


def contract_markers() -> tuple[str, ...]:
    """Unicode and ASCII prompts for each model, plus both legacy prompts.

    ``_engine_prompt`` with ``caps`` omitted is the Unicode form. The ASCII
    form is what the console prints when stdout cannot encode ``›``.
    """
    from fx1.interactive.console import _LEGACY_PROMPT, _engine_prompt
    from fx1.interactive.profiles import DEFAULT_MODEL, MODELS
    from fx1.interactive.terminal import TermCaps

    ascii_caps = TermCaps(True, True, True, False, False, 80, 24, False)
    found = [_LEGACY_PROMPT, "dip > "]
    seen = set(found)
    for model in (*MODELS, DEFAULT_MODEL):
        stub = _ModelStub(model)
        for text in (_engine_prompt(stub), _engine_prompt(stub, ascii_caps)):
            if text not in seen:
                seen.add(text)
                found.append(text)
    return tuple(found)


def prompt_pattern() -> str:
    """Regex for the console prompts this process should wait for."""
    return prompt_pattern_from(contract_markers())


def _arm(pattern: str | None = None, sleep: SleepFn | None = None) -> str:
    text = pattern if pattern else _pattern_or_retry(sleep or time.sleep)
    _ARMED["text"] = text
    _ARMED["compiled"] = re.compile(text.encode("utf-8"))
    return text


def _pattern_or_retry(sleep: SleepFn) -> str:
    try:
        return prompt_pattern()
    except (ImportError, SyntaxError):
        sleep(TRANSIENT_RETRY_S)
        return prompt_pattern()


def _armed_compiled() -> re.Pattern[bytes]:
    compiled = _ARMED.get("compiled")
    if isinstance(compiled, re.Pattern):
        return compiled
    _arm()
    compiled = _ARMED["compiled"]
    if not isinstance(compiled, re.Pattern):
        raise LatencyError("prompt pattern was not compiled")
    return compiled


def _saw_prompt(buf: bytes | bytearray) -> bool:
    return _armed_compiled().search(buf) is not None


def nearest_rank(samples: Sequence[float], percent: float) -> float:
    """Inclusive nearest-rank percentile. Does not mutate ``samples``.

    Rank is ``ceil(percent/100 * n)`` into the sorted sample, clamped to
    ``1..n``. p95 of three samples is the maximum.
    """
    _check_percent(percent)
    ordered = sorted(float(item) for item in samples)
    if not ordered:
        raise ValueError("nearest_rank requires at least one sample")
    return _rank_sorted(ordered, percent)


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """min, median, p50, p95, max, and ``n``. Empty input yields null stats."""
    ordered = sorted(float(item) for item in samples)
    if not ordered:
        return _empty_summary()
    return {
        "n": len(ordered),
        "min_s": _round(ordered[0]),
        "median_s": _round(_median(ordered)),
        "p50_s": _round(_rank_sorted(ordered, 50)),
        "p95_s": _round(_rank_sorted(ordered, 95)),
        "max_s": _round(ordered[-1]),
    }


def evaluate_target(
    observed_s: float | None,
    sample_size: int,
    limit_s: float,
    required_samples: int,
    *,
    failures: int,
) -> dict[str, Any]:
    """Score one proposed limit. Pass needs the number, the sample, and no failures."""
    numeric = observed_s is not None and observed_s <= limit_s
    sample_ok = sample_size >= required_samples and failures == 0
    return {
        "limit_s": limit_s,
        "required_samples": required_samples,
        "observed_s": None if observed_s is None else _round(observed_s),
        "sample_size": sample_size,
        "failures": failures,
        "numeric_pass": numeric,
        "sample_pass": sample_ok,
        "pass": bool(numeric and sample_ok),
    }


def evaluate_targets(
    cold_p95_s: float | None,
    cold_n: int,
    cold_failures: int,
    input_p95_s: float | None,
    input_n: int,
    input_failures: int,
    cancel_s: float | None,
    cancel_ok: bool,
) -> dict[str, Any]:
    """The three proposed P5 numeric targets. Sample floors are the contract's."""
    return {
        "cold_prompt_p95_s": evaluate_target(
            cold_p95_s,
            cold_n,
            COLD_P95_LIMIT_S,
            COLD_REQUIRED_SAMPLES,
            failures=cold_failures,
        ),
        "input_ack_p95_s": evaluate_target(
            input_p95_s,
            input_n,
            INPUT_P95_LIMIT_S,
            INPUT_REQUIRED_SAMPLES,
            failures=input_failures,
        ),
        "cancel_ack_s": _cancel_target(cancel_s, cancel_ok),
    }


def parse_importtime(text: str) -> list[dict[str, Any]]:
    """Parse ``python -X importtime`` stderr. One row per module, max self time."""
    found: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for line in text.splitlines():
        _take_import_line(line, found, order)
    return [_import_row(name, found[name]) for name in order]


def top_import_contributors(
    rows: Sequence[Mapping[str, Any]],
    limit: int = IMPORT_TOP_N,
) -> list[dict[str, Any]]:
    """Highest exclusive self time first. Ties break on cumulative time, then name."""
    ordered = sorted(rows, key=_import_sort_key)
    return [dict(row) for row in ordered[: max(0, limit)]]


def format_import_profile(rows: Sequence[Mapping[str, Any]], *, total: int) -> str:
    """Human table of the top import-time contributors."""
    header = (
        f"import-time top {len(rows)} of {total} modules "
        "(exclusive self_us; python -X importtime until the first chat prompt)"
    )
    lines = [header, "rank  self_us  cumulative_us  occurrences  module"]
    lines.extend(_profile_line(index, row) for index, row in enumerate(rows, start=1))
    return "\n".join(lines) + "\n"


def child_env(
    base: Mapping[str, str],
    config_dir: str,
    session_dir: str,
) -> dict[str, str]:
    """Child environment with an unconfigured endpoint and a private config dir."""
    cleaned = {key: value for key, value in base.items() if not _drop_env(key)}
    src = str(_SRC)
    cleaned["PYTHONPATH"] = _prepend_path(src, cleaned.get("PYTHONPATH"))
    cleaned["FX1_CONFIG_DIR"] = config_dir
    cleaned["FX1_SESSION_DIR"] = session_dir
    cleaned["PYTHONUNBUFFERED"] = "1"
    cleaned["PYTHONIOENCODING"] = "utf-8"
    cleaned["NO_COLOR"] = "1"
    cleaned["OMP_NUM_THREADS"] = "1"
    cleaned["MKL_NUM_THREADS"] = "1"
    cleaned["TOKENIZERS_PARALLELISM"] = "false"
    cleaned.setdefault("TERM", "xterm")
    return cleaned


def environment_facts() -> dict[str, Any]:
    """Platform facts recorded with a measurement. No environment values except the hash seed."""
    seed = os.environ.get("PYTHONHASHSEED")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": _load_average(),
        "pythonhashseed": seed,
    }


def scrub_failure(text: str) -> str:
    """Truncate operator-facing failure text and strip credential shapes."""
    cleaned = _local_scrub(text)
    with contextlib.suppress(ImportError):
        cleaned = _redact_text()(cleaned)
    return cleaned[:_FAILURE_LIMIT]


def build_document(
    *,
    cold: Mapping[str, Any],
    input_ack: Mapping[str, Any],
    cancel: Mapping[str, Any],
    environment: Mapping[str, Any],
    reference_hardware: bool,
    measured_at: str | None = None,
) -> dict[str, Any]:
    """Assemble ``dipcatcher.latency.v1``. ``p5_gate_closed`` is always false."""
    cold_section = _cold_section(cold)
    input_section = _input_section(input_ack)
    cancel_section = _cancel_section(cancel)
    targets = _targets_for(cold_section, input_section, cancel_section)
    passed = all(bool(item["pass"]) for item in targets.values())
    return {
        "schema": SCHEMA,
        "measured_at": measured_at or _measured_at(),
        "prompt_pattern": _pattern_text(cold),
        "reference_hardware": bool(reference_hardware),
        "p5_gate_closed": False,
        "percentile_method": PERCENTILE_METHOD,
        "note": NOTE,
        "environment": dict(environment),
        "cold_prompt": cold_section,
        "input_ack": input_section,
        "cancel_ack": cancel_section,
        "targets": targets,
        "pass": passed,
    }


def dumps_document(document: Mapping[str, Any]) -> str:
    """One UTF-8 JSON document, sorted keys, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_document(
    path: Path,
    document: Mapping[str, Any],
    *,
    sleep: SleepFn = time.sleep,
) -> None:
    """Atomically replace ``path`` at mode 0600 via ``atomicstore``."""
    writer = _load_json_writer(sleep)
    writer(Path(path), dict(document))


def chat_argv(python: str) -> list[str]:
    """Timing-run argv. Does not pass ``-X importtime``."""
    return [python, "-m", "dipcatcher_cli", "chat"]


def measure_cold_runs(
    runs: int,
    timeout_s: float,
    python: str,
    env: Mapping[str, str],
    *,
    sleep: SleepFn = time.sleep,
) -> dict[str, Any]:
    """Fresh-process cold prompt samples. One import-error retry for the whole call."""
    argv = chat_argv(python)
    if not pty_supported():
        return _cold_blocked(argv, runs)
    text = _arm(sleep=sleep)
    samples: list[float] = []
    failures: list[dict[str, Any]] = []
    retry = _Retry()
    for index in range(runs):
        _record_cold(index, argv, env, timeout_s, sleep, retry, samples, failures)
    return {"command": argv, "samples_s": samples, "failures": failures, "prompt_pattern": text}


def measure_input_events(
    events: int,
    event_timeout_s: float,
    prompt_timeout_s: float,
    python: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Slash-command acknowledgement on one already-started chat."""
    cycle = list(LOCAL_COMMANDS)
    if not pty_supported():
        return _input_blocked(events, cycle)
    _arm()
    chat = PtyChat(chat_argv(python), env)
    try:
        return _collect_events(chat, events, event_timeout_s, prompt_timeout_s, cycle)
    finally:
        chat.close()


def measure_cancel(
    python: str,
    env: Mapping[str, str],
    *,
    timeout_s: float,
    sleep: SleepFn = time.sleep,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Time from cancel to sleeper stop through ``run_owned``."""
    argv = [python, str(SCRIPT_PATH), "--sleeper"]
    try:
        invoke = runner if runner is not None else _load_runner(sleep)
    except ImportError as exc:
        return _cancel_error(argv, scrub_failure(str(exc)))
    return _time_cancel(argv, env, timeout_s, invoke)


def collect(
    request: MeasureRequest,
    *,
    cold_fn: ColdFn | None = None,
    event_fn: ColdFn | None = None,
    cancel_fn: ColdFn | None = None,
    facts_fn: FactsFn | None = None,
    sleep: SleepFn | None = None,
    measured_at: str | None = None,
) -> dict[str, Any]:
    """Run or accept the three measurements and return the JSON document."""
    pause = sleep or time.sleep
    facts = facts_fn() if facts_fn is not None else environment_facts()
    env, cleanup = _env_for(request, isolate=_needs_child(cold_fn, event_fn, cancel_fn))
    try:
        cold = _call_cold(request, env, cold_fn, pause)
        events = _call_events(request, env, event_fn)
        cancel = _call_cancel(request, env, cancel_fn, pause)
    finally:
        cleanup()
    return build_document(
        cold=cold,
        input_ack=events,
        cancel=cancel,
        environment=facts,
        reference_hardware=request.reference_hardware,
        measured_at=measured_at,
    )


def capture_importtime(python: str, env: Mapping[str, str], timeout_s: float) -> str:
    """``-X importtime`` stderr for chat-until-prompt. Not used by timing runs."""
    if not pty_supported():
        raise LatencyError(POSIX_PTY_ERROR)
    _arm()
    import pty

    master, slave = pty.openpty()
    err_r, err_w = os.pipe()
    proc: subprocess.Popen[bytes] | None = None
    closed = _Closed()
    try:
        argv = [python, "-X", "importtime", "-m", "dipcatcher_cli", "chat"]
        proc = _spawn_chat(argv, env, slave, err_w)
        _close_quiet(slave)
        closed.slave = True
        _close_quiet(err_w)
        closed.err_w = True
        return _collect_import_stderr(master, err_r, proc, timeout_s)
    finally:
        _release_import(master, slave, err_r, err_w, proc, closed)


def main(
    argv: Sequence[str] | None = None,
    *,
    cold_fn: ColdFn | None = None,
    event_fn: ColdFn | None = None,
    cancel_fn: ColdFn | None = None,
    import_fn: ImportFn | None = None,
    facts_fn: FactsFn | None = None,
    sleep: SleepFn | None = None,
) -> int:
    """CLI entry. ``--sleeper`` blocks until SIGTERM and imports nothing else."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw == ["--sleeper"]:
        return _sleeper()
    try:
        options = _options(raw)
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SystemExit as exc:
        return _system_exit_code(exc)
    if options.import_profile:
        return _main_profile(options.request, import_fn, sleep)
    try:
        document = collect(
            options.request,
            cold_fn=cold_fn,
            event_fn=event_fn,
            cancel_fn=cancel_fn,
            facts_fn=facts_fn,
            sleep=sleep,
        )
        _emit_document(options.out, document, sleep or time.sleep)
    except (OSError, LatencyError) as exc:
        print(scrub_failure(str(exc)), file=sys.stderr)
        return 1
    print(format_summary(document), file=sys.stderr)
    return 0 if document["pass"] else 1


def format_summary(document: Mapping[str, Any]) -> str:
    """One secret-free status line."""
    cold = document["cold_prompt"]
    ack = document["input_ack"]
    cancel = document["cancel_ack"]
    return (
        f"pass={str(bool(document['pass'])).lower()} "
        f"reference_hardware={str(bool(document['reference_hardware'])).lower()} "
        f"p5_gate_closed=false "
        f"cold_p95_s={cold.get('p95_s')} "
        f"input_p95_s={ack.get('p95_s')} "
        f"cancel_s={cancel.get('seconds')} "
        f"cold_n={cold.get('n_ok')} "
        f"input_n={ack.get('n_ok')}"
    )


class PtyChat:
    """One chat process on a POSIX pseudo-terminal."""

    def __init__(self, argv: Sequence[str], env: Mapping[str, str]) -> None:
        self._argv = [str(part) for part in argv]
        self._env = {str(key): str(value) for key, value in env.items()}
        self._pattern = _armed_compiled()
        self._master: int | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._tail = bytearray()

    def start(self) -> None:
        if not pty_supported():
            raise LatencyError(POSIX_PTY_ERROR)
        import pty

        master, slave = pty.openpty()
        self._master = master
        try:
            self._proc = _spawn_chat(self._argv, self._env, slave, slave)
        finally:
            _close_quiet(slave)
            if self._proc is None:
                self.close()

    def read_until(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        buf = bytearray()
        while self._pattern.search(buf) is None:
            buf.extend(self._read_chunk(deadline))

    def write_line(self, command: str) -> None:
        os.write(self._master_fd(), command.encode("utf-8") + b"\r")

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            _stop_process(proc)
        master = self._master
        self._master = None
        if master is not None:
            _close_quiet(master)

    def _read_chunk(self, deadline: float) -> bytes:
        master = self._master_fd()
        if deadline <= time.perf_counter():
            raise TimeoutError(self._error_text("timed out waiting for the chat prompt"))
        if not _select([master], deadline - time.perf_counter()):
            raise TimeoutError(self._error_text("timed out waiting for the chat prompt"))
        chunk = self._read_ready(master)
        if chunk == b"":
            raise EOFError(self._error_text("chat process closed its output"))
        self._remember(chunk)
        return chunk

    def _read_ready(self, master: int) -> bytes:
        try:
            return os.read(master, _READ_SIZE)
        except OSError as exc:
            detail = exc.strerror or "read failed"
            raise EOFError(self._error_text(detail)) from exc

    def _master_fd(self) -> int:
        if self._master is None:
            raise LatencyError("PTY is closed")
        return self._master

    def _remember(self, chunk: bytes) -> None:
        self._tail.extend(chunk)
        if len(self._tail) > _TAIL_LIMIT:
            del self._tail[:-_TAIL_LIMIT]

    def _error_text(self, prefix: str) -> str:
        tail = self._tail.decode("utf-8", "replace")
        return scrub_failure(f"{prefix}; tail={tail}")


class _Closed:
    slave: bool = False
    err_w: bool = False


def _sleeper() -> int:
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    time.sleep(3600)
    return 0


def _check_percent(percent: float) -> None:
    if percent <= 0 or percent > 100:
        raise ValueError("percent must be in (0, 100]")


def _rank_sorted(ordered: Sequence[float], percent: float) -> float:
    rank = math.ceil((percent / 100.0) * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return float(ordered[index])


def _median(ordered: Sequence[float]) -> float:
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _empty_summary() -> dict[str, Any]:
    return {
        "n": 0,
        "min_s": None,
        "median_s": None,
        "p50_s": None,
        "p95_s": None,
        "max_s": None,
    }


def _round(value: float) -> float:
    return round(float(value), 6)


def _cancel_target(cancel_s: float | None, cancel_ok: bool) -> dict[str, Any]:
    observed = cancel_s if cancel_ok else None
    return evaluate_target(
        observed,
        1 if cancel_ok else 0,
        CANCEL_LIMIT_S,
        CANCEL_REQUIRED_SAMPLES,
        failures=0 if cancel_ok else 1,
    )


def _take_import_line(
    line: str,
    found: dict[str, dict[str, int]],
    order: list[str],
) -> None:
    match = _IMPORT_LINE.match(line)
    if match is None:
        return
    name = match.group(3).strip()
    if not name:
        return
    self_us = int(match.group(1))
    cumulative = int(match.group(2))
    _store_import(found, order, name, self_us, cumulative)


def _store_import(
    found: dict[str, dict[str, int]],
    order: list[str],
    name: str,
    self_us: int,
    cumulative: int,
) -> None:
    current = found.get(name)
    if current is None:
        found[name] = {"self_us": self_us, "cumulative_us": cumulative, "occurrences": 1}
        order.append(name)
        return
    current["occurrences"] += 1
    current["self_us"] = max(current["self_us"], self_us)
    current["cumulative_us"] = max(current["cumulative_us"], cumulative)


def _import_row(name: str, raw: Mapping[str, int]) -> dict[str, Any]:
    return {
        "module": name,
        "self_us": raw["self_us"],
        "cumulative_us": raw["cumulative_us"],
        "occurrences": raw["occurrences"],
    }


def _import_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (-int(row["self_us"]), -int(row["cumulative_us"]), str(row["module"]))


def _profile_line(index: int, row: Mapping[str, Any]) -> str:
    return (
        f"{index:>4}  {int(row['self_us']):>10}  {int(row['cumulative_us']):>13}  "
        f"{int(row['occurrences']):>11}  {row['module']}"
    )


def _drop_env(key: str) -> bool:
    folded = key.casefold()
    if folded in _DROP_EXACT or "password" in folded:
        return True
    return folded.endswith(_DROP_SUFFIXES)


def _prepend_path(entry: str, existing: str | None) -> str:
    if not existing:
        return entry
    parts = existing.split(os.pathsep)
    if entry in parts:
        return existing
    return entry + os.pathsep + existing


def _load_average() -> list[float] | None:
    reader: Callable[[], tuple[float, float, float]] | None
    reader = getattr(os, "getloadavg", None)
    if reader is None:
        return None
    try:
        sample = reader()
    except OSError:
        return None
    return [round(float(part), 4) for part in sample]


def _local_scrub(text: str) -> str:
    cleaned = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", text)
    return re.sub(
        r"(?i)(api[_-]?key\s*[=:]\s*)(\S+)",
        r"\1[REDACTED]",
        cleaned,
    )


def _redact_text() -> Callable[[str], str]:
    module = _load_module("dipcatcher_latency_redact", "redact.py")
    fn: Callable[[str], str] = module.redact_text
    return fn


def _measured_at() -> str:
    return datetime.now(UTC).isoformat()


def _cold_section(raw: Mapping[str, Any]) -> dict[str, Any]:
    samples, stats = _rounded_stats(raw.get("samples_s", []))
    failures = _failure_list(raw.get("failures", []))
    attempted = _attempted(raw, len(samples), len(failures))
    return {
        "command": [str(part) for part in raw.get("command", [])],
        "prompt_pattern": _pattern_text(raw),
        "n_attempted": attempted,
        "n_ok": stats["n"],
        "min_s": stats["min_s"],
        "median_s": stats["median_s"],
        "p95_s": stats["p95_s"],
        "max_s": stats["max_s"],
        "samples_s": samples,
        "failures": failures,
    }


def _input_section(raw: Mapping[str, Any]) -> dict[str, Any]:
    samples, stats = _rounded_stats(raw.get("samples_s", []))
    failures = _failure_list(raw.get("failures", []))
    cycle = raw.get("command_cycle", list(LOCAL_COMMANDS))
    return {
        "command_cycle": [str(item) for item in cycle],
        "n_attempted": _attempted(raw, len(samples), len(failures)),
        "n_ok": stats["n"],
        "p50_s": stats["p50_s"],
        "p95_s": stats["p95_s"],
        "min_s": stats["min_s"],
        "max_s": stats["max_s"],
        "samples_s": samples,
        "failures": failures,
    }


def _cancel_section(raw: Mapping[str, Any]) -> dict[str, Any]:
    seconds = _optional_float(raw.get("seconds"))
    stopped = bool(raw.get("stopped"))
    error = raw.get("error")
    owned_pass = bool(stopped and seconds is not None and seconds <= OWNED_STOP_LIMIT_S)
    return {
        "method": str(raw.get("method") or CANCEL_METHOD),
        "chat_cancel": bool(raw.get("chat_cancel")),
        "reason": str(raw.get("reason") or CANCEL_REASON),
        "argv": [str(part) for part in raw.get("argv", [])],
        "cancel_poll_s": _optional_float(raw.get("cancel_poll_s")) or CANCEL_POLL_S,
        "seconds": None if seconds is None else _round(seconds),
        "stopped": stopped,
        "cancelled": bool(raw.get("cancelled")),
        "exit_code": raw.get("exit_code"),
        "owned_stop_limit_s": OWNED_STOP_LIMIT_S,
        "owned_stop_pass": owned_pass,
        "error": None if error is None else scrub_failure(str(error)),
    }


def _targets_for(
    cold: Mapping[str, Any],
    ack: Mapping[str, Any],
    cancel: Mapping[str, Any],
) -> dict[str, Any]:
    cancel_ok = bool(
        cancel.get("stopped") and cancel.get("cancelled") and cancel.get("error") is None
    )
    seconds = cancel.get("seconds")
    observed = float(seconds) if cancel_ok and seconds is not None else None
    return evaluate_targets(
        _optional_float(cold.get("p95_s")),
        int(cold.get("n_ok") or 0),
        len(cold.get("failures") or []),
        _optional_float(ack.get("p95_s")),
        int(ack.get("n_ok") or 0),
        len(ack.get("failures") or []),
        observed,
        cancel_ok,
    )


def _rounded_stats(samples: object) -> tuple[list[float], dict[str, Any]]:
    rounded = [_round(float(item)) for item in _as_list(samples)]
    return rounded, summarize(rounded)


def _failure_list(raw: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if isinstance(item, Mapping):
            rows.append({str(key): _public_value(item[key]) for key in item})
    return rows


def _public_value(value: object) -> object:
    if isinstance(value, str):
        return scrub_failure(value)
    return value


def _attempted(raw: Mapping[str, Any], ok: int, failed: int) -> int:
    if "n_attempted" in raw:
        return int(raw["n_attempted"])
    return ok + failed


def _as_list(raw: object) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pattern_text(raw: Mapping[str, Any]) -> str:
    value = raw.get("prompt_pattern")
    if isinstance(value, str) and value:
        return value
    armed = _ARMED.get("text")
    if isinstance(armed, str) and armed:
        return armed
    return FALLBACK_PROMPT_PATTERN


def _cold_blocked(argv: Sequence[str], runs: int) -> dict[str, Any]:
    failures = [{"run": index, "error": POSIX_PTY_ERROR} for index in range(runs)]
    return {"command": list(argv), "samples_s": [], "failures": failures, "n_attempted": runs}


def _record_cold(
    index: int,
    argv: Sequence[str],
    env: Mapping[str, str],
    timeout_s: float,
    sleep: SleepFn,
    retry: _Retry,
    samples: list[float],
    failures: list[dict[str, Any]],
) -> None:
    elapsed, error = _time_to_prompt(argv, env, timeout_s)
    if elapsed is None and _should_retry(error, retry):
        retry.used = True
        sleep(TRANSIENT_RETRY_S)
        elapsed, error = _time_to_prompt(argv, env, timeout_s)
    if elapsed is None:
        failures.append({"run": index, "error": error or "cold prompt was not observed"})
        return
    samples.append(elapsed)


def _should_retry(error: str | None, retry: _Retry) -> bool:
    if retry.used or error is None:
        return False
    markers = ("ImportError", "ModuleNotFoundError", "SyntaxError")
    return any(marker in error for marker in markers)


def _time_to_prompt(
    argv: Sequence[str],
    env: Mapping[str, str],
    timeout_s: float,
) -> tuple[float | None, str | None]:
    chat = PtyChat(argv, env)
    started = time.perf_counter()
    try:
        chat.start()
        chat.read_until(timeout_s)
    except (TimeoutError, EOFError, OSError, LatencyError) as exc:
        return None, scrub_failure(str(exc))
    else:
        return time.perf_counter() - started, None
    finally:
        chat.close()


def _input_blocked(events: int, cycle: Sequence[str]) -> dict[str, Any]:
    failures = [{"event": index, "error": POSIX_PTY_ERROR} for index in range(events)]
    return _input_body(events, cycle, [], failures)


def _collect_events(
    chat: PtyChat,
    events: int,
    event_timeout_s: float,
    prompt_timeout_s: float,
    cycle: Sequence[str],
) -> dict[str, Any]:
    try:
        chat.start()
        chat.read_until(prompt_timeout_s)
    except (TimeoutError, EOFError, OSError, LatencyError) as exc:
        return _input_failed(events, cycle, scrub_failure(str(exc)))
    return _run_events(chat, events, event_timeout_s, cycle)


def _run_events(
    chat: PtyChat,
    events: int,
    timeout_s: float,
    cycle: Sequence[str],
) -> dict[str, Any]:
    samples: list[float] = []
    failures: list[dict[str, Any]] = []
    for index in range(events):
        if not _one_event(chat, index, timeout_s, cycle, samples, failures):
            break
    return _input_body(events, cycle, samples, failures)


def _one_event(
    chat: PtyChat,
    index: int,
    timeout_s: float,
    cycle: Sequence[str],
    samples: list[float],
    failures: list[dict[str, Any]],
) -> bool:
    command = cycle[index % len(cycle)]
    started = time.perf_counter()
    try:
        chat.write_line(command)
        chat.read_until(timeout_s)
    except (TimeoutError, EOFError, OSError) as exc:
        failures.append({"event": index, "command": command, "error": scrub_failure(str(exc))})
        return False
    samples.append(time.perf_counter() - started)
    return True


def _input_failed(events: int, cycle: Sequence[str], error: str) -> dict[str, Any]:
    return _input_body(events, cycle, [], [{"event": 0, "error": error}])


def _input_body(
    events: int,
    cycle: Sequence[str],
    samples: list[float],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "command_cycle": list(cycle),
        "n_attempted": events,
        "samples_s": samples,
        "failures": failures,
    }


def _cancel_error(argv: Sequence[str], error: str) -> dict[str, Any]:
    return _cancel_body(
        argv, seconds=None, stopped=False, cancelled=False, exit_code=None, error=error
    )


def _time_cancel(
    argv: Sequence[str],
    env: Mapping[str, str],
    timeout_s: float,
    runner: Runner,
) -> dict[str, Any]:
    cancel_event = threading.Event()
    spawned = threading.Event()
    holder: dict[str, Any] = {}

    def emit(name: str, fields: Mapping[str, Any]) -> None:
        del fields
        if name == "proc.spawn":
            spawned.set()

    worker = _CancelWorker(runner, argv, env, timeout_s, cancel_event, emit, holder)
    thread = threading.Thread(target=worker, name="dipcatcher-latency-cancel")
    thread.start()
    if not spawned.wait(timeout_s):
        return _finish_cancel_wait(thread, cancel_event, timeout_s, argv, holder, spawned=False)
    return _finish_cancel_wait(thread, cancel_event, timeout_s, argv, holder, spawned=True)


class _CancelWorker:
    """One owned sleeper. A class keeps the thread target's complexity small."""

    def __init__(
        self,
        runner: Runner,
        argv: Sequence[str],
        env: Mapping[str, str],
        timeout_s: float,
        cancel_event: Any,
        emit: Callable[[str, Mapping[str, Any]], None],
        holder: dict[str, Any],
    ) -> None:
        self._runner = runner
        self._argv = list(argv)
        self._env = dict(env)
        self._timeout_s = timeout_s
        self._cancel_event = cancel_event
        self._emit = emit
        self._holder = holder

    def __call__(self) -> None:
        try:
            self._holder["result"] = self._runner(
                self._argv,
                cancel_event=self._cancel_event,
                env=self._env,
                emit=self._emit,
                timeout_s=self._timeout_s + 30.0,
            )
        except Exception as exc:  # noqa: BLE001 — record and let the joiner score it
            self._holder["error"] = scrub_failure(type(exc).__name__)


def _finish_cancel_wait(
    thread: Any,
    cancel_event: Any,
    timeout_s: float,
    argv: Sequence[str],
    holder: dict[str, Any],
    *,
    spawned: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    cancel_event.set()
    thread.join(timeout_s)
    elapsed = time.perf_counter() - started
    if not spawned:
        return _cancel_error(argv, "sleeper did not spawn before timeout")
    if thread.is_alive():
        return _cancel_error(argv, "cancel did not stop the sleeper before timeout")
    if "error" in holder:
        return _cancel_error(argv, str(holder["error"]))
    return _cancel_from_result(argv, holder.get("result"), elapsed)


def _cancel_from_result(argv: Sequence[str], result: Any, elapsed: float) -> dict[str, Any]:
    if result is None:
        return _cancel_error(argv, "owned runner returned no result")
    spawn_error = getattr(result, "spawn_error", None)
    if spawn_error:
        return _cancel_error(argv, scrub_failure(str(spawn_error)))
    return _cancel_body(
        argv,
        seconds=elapsed,
        stopped=True,
        cancelled=bool(getattr(result, "cancelled", False)),
        exit_code=getattr(result, "exit_code", None),
        error=None,
    )


def _cancel_body(
    argv: Sequence[str],
    *,
    seconds: float | None,
    stopped: bool,
    cancelled: bool,
    exit_code: object,
    error: str | None,
) -> dict[str, Any]:
    return {
        "method": CANCEL_METHOD,
        "chat_cancel": False,
        "reason": CANCEL_REASON,
        "argv": list(argv),
        "cancel_poll_s": CANCEL_POLL_S,
        "seconds": seconds,
        "stopped": stopped,
        "cancelled": cancelled,
        "exit_code": exit_code,
        "error": error,
    }


def _load_runner(sleep: SleepFn) -> Runner:
    try:
        return _runner_from_procs()
    except (ImportError, SyntaxError):
        sleep(TRANSIENT_RETRY_S)
        return _runner_from_procs()


def _runner_from_procs() -> Runner:
    module = _load_package_module("fx1.interactive.procs")
    limits_cls = module.ProcLimits
    registry_cls = module.ProcessRegistry

    def runner(
        argv: Sequence[str],
        *,
        cancel_event: Any,
        env: Mapping[str, str],
        emit: Callable[[str, Mapping[str, Any]], None],
        timeout_s: float,
    ) -> Any:
        limits = limits_cls(timeout_s=timeout_s, cancel_poll_s=CANCEL_POLL_S, kill_grace_s=2.0)
        return module.run_owned(
            list(argv),
            limits,
            env=dict(env),
            cancel_event=cancel_event,
            registry=registry_cls(),
            emit=emit,
        )

    return runner


def _load_package_module(name: str) -> Any:
    """Import a submodule. Retries belong to the caller; this raises on a bad tree."""
    return importlib.import_module(name)


def _needs_child(
    cold_fn: ColdFn | None,
    event_fn: ColdFn | None,
    cancel_fn: ColdFn | None,
) -> bool:
    return cold_fn is None or event_fn is None or cancel_fn is None


def _env_for(
    request: MeasureRequest, *, isolate: bool
) -> tuple[dict[str, str], Callable[[], None]]:
    if not isolate:
        return {}, _noop
    root = tempfile.TemporaryDirectory(prefix="dipcatcher-latency-")
    config = str(Path(root.name) / "config")
    session = str(Path(root.name) / "sessions")
    Path(config).mkdir(parents=True, exist_ok=True)
    Path(session).mkdir(parents=True, exist_ok=True)
    return child_env(os.environ, config, session), root.cleanup


def _noop() -> None:
    return None


def _python_of(request: MeasureRequest) -> str:
    return request.python or sys.executable


def _call_cold(
    request: MeasureRequest,
    env: Mapping[str, str],
    cold_fn: ColdFn | None,
    sleep: SleepFn,
) -> Mapping[str, Any]:
    if cold_fn is not None:
        return cold_fn(request, env)
    return measure_cold_runs(
        request.runs, request.cold_timeout_s, _python_of(request), env, sleep=sleep
    )


def _call_events(
    request: MeasureRequest,
    env: Mapping[str, str],
    event_fn: ColdFn | None,
) -> Mapping[str, Any]:
    if event_fn is not None:
        return event_fn(request, env)
    return measure_input_events(
        request.events,
        request.event_timeout_s,
        request.cold_timeout_s,
        _python_of(request),
        env,
    )


def _call_cancel(
    request: MeasureRequest,
    env: Mapping[str, str],
    cancel_fn: ColdFn | None,
    sleep: SleepFn,
) -> Mapping[str, Any]:
    if cancel_fn is not None:
        return cancel_fn(request, env)
    return measure_cancel(_python_of(request), env, timeout_s=request.cancel_timeout_s, sleep=sleep)


def _spawn_chat(
    argv: Sequence[str],
    env: Mapping[str, str],
    slave: int,
    stderr: int,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(argv),
        stdin=slave,
        stdout=slave,
        stderr=stderr,
        env={str(key): str(value) for key, value in env.items()},
        cwd=str(REPO_ROOT),
        start_new_session=True,
        close_fds=True,
    )


def _collect_import_stderr(
    master: int,
    err_r: int,
    proc: subprocess.Popen[bytes],
    timeout_s: float,
) -> str:
    os.set_blocking(master, False)
    os.set_blocking(err_r, False)
    out = bytearray()
    err = bytearray()
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        _pump(master, err_r, out, err, deadline)
        if _saw_prompt(out):
            _drain(err_r, err)
            return bytes(err).decode("utf-8", "replace")
        if proc.poll() is not None:
            break
    _drain(master, out)
    _drain(err_r, err)
    if err and _saw_prompt(out):
        return bytes(err).decode("utf-8", "replace")
    detail = scrub_failure(out.decode("utf-8", "replace"))
    raise TimeoutError(detail or "import profile did not reach the chat prompt")


def _pump(
    master: int,
    err_r: int,
    out: bytearray,
    err: bytearray,
    deadline: float,
) -> None:
    remaining = max(0.0, deadline - time.perf_counter())
    ready = _select([master, err_r], min(0.2, remaining))
    if master in ready:
        _drain(master, out)
    if err_r in ready:
        _drain(err_r, err)


def _drain(fd: int, sink: bytearray) -> None:
    while len(sink) < _IMPORT_CAP:
        try:
            chunk = os.read(fd, _READ_SIZE)
        except BlockingIOError:
            return
        if not chunk:
            return
        sink.extend(chunk)


def _release_import(
    master: int,
    slave: int,
    err_r: int,
    err_w: int,
    proc: subprocess.Popen[bytes] | None,
    closed: _Closed,
) -> None:
    if not closed.slave:
        _close_quiet(slave)
    if not closed.err_w:
        _close_quiet(err_w)
    if proc is not None:
        _stop_process(proc)
    _close_quiet(master)
    _close_quiet(err_r)


def _close_quiet(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        return


def _select(fds: Sequence[int], timeout_s: float) -> list[int]:
    while True:
        try:
            ready, _, _ = select.select(list(fds), [], [], max(0.0, timeout_s))
        except InterruptedError:
            continue
        return list(ready)


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        _reap(proc)
        return
    _signal_proc(proc, signal.SIGTERM)
    if _wait_exit(proc, 2.0):
        return
    _signal_proc(proc, _sigkill())
    _wait_exit(proc, 2.0)


def _reap(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        return


def _signal_proc(proc: subprocess.Popen[bytes], sig: int) -> None:
    if os.name == "posix":
        _signal_group(proc, sig)
        return
    if sig == signal.SIGTERM:
        proc.terminate()
        return
    proc.kill()


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except OSError:
        try:
            proc.send_signal(sig)
        except OSError:
            return


def _wait_exit(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return True


def _sigkill() -> int:
    value = getattr(signal, "SIGKILL", signal.SIGTERM)
    return int(value)


def _options(argv: Sequence[str]) -> _Options:
    parser = _parser()
    namespace = parser.parse_args(list(argv))
    _validate(namespace)
    return _Options(
        request=_request_from(namespace),
        out=namespace.out,
        import_profile=bool(namespace.import_profile),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dipcatcher_latency",
        description=(
            "Measure dipcatcher chat cold-prompt time, local slash-command "
            "acknowledgement, and owned-sleeper cancel-to-stop. "
            "Proposed P5 targets are scored on this machine and are not a "
            "reference-hardware certification."
        ),
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--events", type=int, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reference-hardware", action="store_true")
    parser.add_argument("--import-profile", action="store_true")
    parser.add_argument("--cold-timeout-s", type=float, default=30.0)
    parser.add_argument("--event-timeout-s", type=float, default=5.0)
    parser.add_argument("--cancel-timeout-s", type=float, default=10.0)
    parser.add_argument("--python", default=sys.executable)
    return parser


def _validate(namespace: argparse.Namespace) -> None:
    _check_count("--runs", namespace.runs, MAX_RUNS)
    _check_count("--events", namespace.events, MAX_EVENTS)
    _check_timeout("--cold-timeout-s", namespace.cold_timeout_s)
    _check_timeout("--event-timeout-s", namespace.event_timeout_s)
    _check_timeout("--cancel-timeout-s", namespace.cancel_timeout_s)


def _check_count(flag: str, value: int, limit: int) -> None:
    if value < 1 or value > limit:
        raise UsageError(f"{flag} must be an integer from 1 to {limit}")


def _check_timeout(flag: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0 or value > MAX_TIMEOUT_S:
        raise UsageError(f"{flag} must be a finite number in (0, {MAX_TIMEOUT_S:g}]")


def _request_from(namespace: argparse.Namespace) -> MeasureRequest:
    return MeasureRequest(
        runs=namespace.runs,
        events=namespace.events,
        reference_hardware=bool(namespace.reference_hardware),
        cold_timeout_s=namespace.cold_timeout_s,
        event_timeout_s=namespace.event_timeout_s,
        cancel_timeout_s=namespace.cancel_timeout_s,
        python=str(namespace.python),
    )


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 2


def _main_profile(
    request: MeasureRequest,
    import_fn: ImportFn | None,
    sleep: SleepFn | None,
) -> int:
    del sleep
    env, cleanup = _env_for(request, isolate=True)
    try:
        text = _profile_text(request, env, import_fn)
    except (LatencyError, OSError, TimeoutError) as exc:
        print(f"import profile failed: {scrub_failure(str(exc))}", file=sys.stderr)
        return 1
    finally:
        cleanup()
    rows = parse_importtime(text)
    top = top_import_contributors(rows, IMPORT_TOP_N)
    if not top:
        print("import profile failed: no import time rows", file=sys.stderr)
        return 1
    sys.stdout.write(format_import_profile(top, total=len(rows)))
    return 0


def _profile_text(
    request: MeasureRequest,
    env: Mapping[str, str],
    import_fn: ImportFn | None,
) -> str:
    if import_fn is not None:
        return import_fn(request, env)
    return capture_importtime(_python_of(request), env, request.cold_timeout_s)


def _emit_document(path: Path | None, document: Mapping[str, Any], sleep: SleepFn) -> None:
    if path is None:
        sys.stdout.write(dumps_document(document))
        return
    write_document(path, document, sleep=sleep)


def _load_json_writer(sleep: SleepFn) -> Callable[[Path, Mapping[str, Any]], None]:
    try:
        return _json_writer()
    except (ImportError, SyntaxError):
        sleep(TRANSIENT_RETRY_S)
        return _json_writer()


def _json_writer() -> Callable[[Path, Mapping[str, Any]], None]:
    module = _load_module("dipcatcher_latency_atomicstore", "atomicstore.py")

    def writer(path: Path, document: Mapping[str, Any]) -> None:
        module.atomic_write_json(path, dict(document))

    return writer


@functools.cache
def _load_module(alias: str, filename: str) -> Any:
    """Load a kernel module by path so ``fx1.interactive``'s package init does not run.

    That package init imports the chat stack. Atomic writes and redaction must
    stay usable while another editor is mid-way through ``concierge.py``.
    """
    path = _SRC / "fx1" / "interactive" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = [
    "CANCEL_LIMIT_S",
    "CANCEL_METHOD",
    "CANCEL_REASON",
    "COLD_P95_LIMIT_S",
    "COLD_REQUIRED_SAMPLES",
    "DEFAULT_EVENTS",
    "DEFAULT_RUNS",
    "IMPORT_TOP_N",
    "INPUT_P95_LIMIT_S",
    "INPUT_REQUIRED_SAMPLES",
    "MeasureRequest",
    "OWNED_STOP_LIMIT_S",
    "FALLBACK_PROMPT_PATTERN",
    "PROMPT",
    "SCHEMA",
    "TRANSIENT_RETRY_S",
    "build_document",
    "capture_importtime",
    "chat_argv",
    "child_env",
    "collect",
    "dumps_document",
    "environment_facts",
    "evaluate_target",
    "evaluate_targets",
    "format_import_profile",
    "format_summary",
    "main",
    "measure_cancel",
    "measure_cold_runs",
    "measure_input_events",
    "nearest_rank",
    "parse_importtime",
    "prompt_pattern",
    "prompt_pattern_from",
    "contract_markers",
    "pty_supported",
    "scrub_failure",
    "summarize",
    "top_import_contributors",
    "write_document",
]


if __name__ == "__main__":
    raise SystemExit(main())
