"""Hermetic tests for the dipcatcher latency tool.

Timing assertions are generous liveness bounds only. Percentiles, schema,
and pass/fail use injected samples.
"""

from __future__ import annotations

import json
import os
import platform
import re
import select
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import dipcatcher_latency as lat

SECRET = "sk-latencyfixturekey123456"
IMPORT_TEXT = """\
import time: self [us] | cumulative | imported package
import time:       100 |        100 | alpha
import time:        50 |        150 |   beta
import time:       400 |        400 | gamma
not a row
import time:        80 |        500 |   beta
"""
ENV = {
    "platform": "test",
    "python": "3.12.0",
    "cpu_count": 4,
    "load_average": [0.1, 0.2, 0.3],
    "pythonhashseed": None,
}
STAMP = "2026-01-01T00:00:00+00:00"


def _cold(
    samples: list[float], failures: list[dict[str, object]] | None = None
) -> dict[str, object]:
    failed = [] if failures is None else failures
    return {
        "command": ["python", "-m", "dipcatcher_cli", "chat"],
        "samples_s": samples,
        "failures": failed,
        "n_attempted": len(samples) + len(failed),
    }


def _events(samples: list[float]) -> dict[str, object]:
    return {
        "command_cycle": ["/help", "/status"],
        "samples_s": samples,
        "failures": [],
        "n_attempted": len(samples),
    }


def _cancel(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "seconds": 0.04,
        "stopped": True,
        "cancelled": True,
        "exit_code": 130,
        "error": None,
        "argv": ["python", "--sleeper"],
    }
    body.update(overrides)
    return body


def _document(
    cold_samples: list[float],
    event_samples: list[float],
    cancel: dict[str, object] | None = None,
    *,
    reference_hardware: bool = False,
) -> dict[str, object]:
    return lat.build_document(
        cold=_cold(cold_samples),
        input_ack=_events(event_samples),
        cancel=_cancel() if cancel is None else cancel,
        environment=ENV,
        reference_hardware=reference_hardware,
        measured_at=STAMP,
    )


def _passing() -> dict[str, object]:
    return _document([0.2] * 30, [0.01] * 200)


class _Proc:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.signaled: list[int] = []

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def send_signal(self, sig: int) -> None:
        self.signaled.append(sig)


def test_limits_match_the_product_contract() -> None:
    assert lat.SCHEMA == "dipcatcher.latency.v1"
    assert lat.COLD_P95_LIMIT_S == 2.0
    assert lat.COLD_REQUIRED_SAMPLES == 30
    assert lat.INPUT_P95_LIMIT_S == 0.1
    assert lat.INPUT_REQUIRED_SAMPLES == 200
    assert lat.CANCEL_LIMIT_S == 1.0
    assert lat.CANCEL_REQUIRED_SAMPLES == 1
    assert lat.OWNED_STOP_LIMIT_S == 5.0
    assert lat.DEFAULT_RUNS == 30
    assert lat.DEFAULT_EVENTS == 200
    assert lat.TRANSIENT_RETRY_S == 30.0
    assert lat.IMPORT_TOP_N == 20


def test_nearest_rank_definition_and_does_not_mutate() -> None:
    data = [3.0, 1.0, 2.0]
    assert lat.nearest_rank(data, 100) == 3.0
    assert data == [3.0, 1.0, 2.0]
    assert lat.nearest_rank([0.1, 0.2, 0.3], 95) == 0.3
    assert lat.nearest_rank([0.2], 95) == 0.2
    samples = [float(item) for item in range(1, 101)]
    assert lat.nearest_rank(samples, 95) == 95.0
    assert lat.nearest_rank(samples, 50) == 50.0
    with pytest.raises(ValueError):
        lat.nearest_rank([], 95)
    with pytest.raises(ValueError):
        lat.nearest_rank([1.0], 0)


def test_summarize_median_differs_from_nearest_rank_p50() -> None:
    stats = lat.summarize([4, 1, 3, 2])
    assert stats == {
        "n": 4,
        "min_s": 1.0,
        "median_s": 2.5,
        "p50_s": 2.0,
        "p95_s": 4.0,
        "max_s": 4.0,
    }
    wide = lat.summarize(range(1, 101))
    assert wide["median_s"] == 50.5
    assert wide["p50_s"] == 50.0
    assert wide["p95_s"] == 95.0
    assert lat.summarize(())["n"] == 0
    assert lat.summarize(())["p95_s"] is None


def test_evaluate_target_boundary_sample_and_failures() -> None:
    exact = lat.evaluate_target(2.0, 30, 2.0, 30, failures=0)
    assert exact["numeric_pass"] is True
    assert exact["sample_pass"] is True
    assert exact["pass"] is True
    over = lat.evaluate_target(2.0001, 30, 2.0, 30, failures=0)
    assert over["numeric_pass"] is False
    assert over["pass"] is False
    short = lat.evaluate_target(0.2, 3, 2.0, 30, failures=0)
    assert short["numeric_pass"] is True
    assert short["sample_pass"] is False
    assert short["pass"] is False
    dirty = lat.evaluate_target(0.2, 30, 2.0, 30, failures=1)
    assert dirty["sample_pass"] is False
    assert dirty["pass"] is False
    missing = lat.evaluate_target(None, 0, 1.0, 1, failures=1)
    assert missing["observed_s"] is None
    assert missing["pass"] is False


def test_cancel_sample_is_ignored_unless_the_stop_was_a_cancel() -> None:
    bad = lat.evaluate_targets(0.2, 30, 0, 0.01, 200, 0, 0.01, False)
    assert bad["cancel_ack_s"]["observed_s"] is None
    assert bad["cancel_ack_s"]["pass"] is False
    good = lat.evaluate_targets(0.2, 30, 0, 0.01, 200, 0, 1.0, True)
    assert good["cancel_ack_s"]["pass"] is True
    assert good["cold_prompt_p95_s"]["limit_s"] == 2.0
    assert good["input_ack_p95_s"]["required_samples"] == 200


def test_document_schema_pass_and_reference_flag() -> None:
    doc = _passing()
    assert set(doc) == {
        "schema",
        "measured_at",
        "prompt_pattern",
        "reference_hardware",
        "p5_gate_closed",
        "percentile_method",
        "note",
        "environment",
        "cold_prompt",
        "input_ack",
        "cancel_ack",
        "targets",
        "pass",
    }
    assert doc["schema"] == lat.SCHEMA
    assert doc["pass"] is True
    assert doc["p5_gate_closed"] is False
    assert doc["reference_hardware"] is False
    assert doc["percentile_method"] == "nearest_rank"
    assert doc["measured_at"] == STAMP
    cold = doc["cold_prompt"]
    assert isinstance(cold, dict)
    assert set(cold) == {
        "command",
        "prompt_pattern",
        "n_attempted",
        "n_ok",
        "min_s",
        "median_s",
        "p95_s",
        "max_s",
        "samples_s",
        "failures",
    }
    assert cold["prompt_pattern"] == doc["prompt_pattern"]
    _assert_prompt_matches_known(str(cold["prompt_pattern"]))
    assert cold["p95_s"] == 0.2
    assert cold["n_ok"] == 30
    ack = doc["input_ack"]
    assert isinstance(ack, dict)
    assert ack["p50_s"] == 0.01
    assert ack["p95_s"] == 0.01
    assert ack["n_ok"] == 200
    targets = doc["targets"]
    assert isinstance(targets, dict)
    assert set(targets) == {"cold_prompt_p95_s", "input_ack_p95_s", "cancel_ack_s"}
    flagged = _document([0.2] * 30, [0.01] * 200, reference_hardware=True)
    assert flagged["reference_hardware"] is True
    assert flagged["p5_gate_closed"] is False
    assert flagged["targets"]["cold_prompt_p95_s"]["limit_s"] == 2.0  # type: ignore[index]


def test_short_sample_is_not_a_pass_and_slow_cancel_splits_the_five_second_clause() -> None:
    short = _document([0.2, 0.1, 0.3], [0.01] * 5)
    assert short["pass"] is False
    cold_target = short["targets"]["cold_prompt_p95_s"]  # type: ignore[index]
    input_target = short["targets"]["input_ack_p95_s"]  # type: ignore[index]
    assert cold_target["numeric_pass"] is True
    assert cold_target["sample_pass"] is False
    assert cold_target["observed_s"] == 0.3
    assert input_target["sample_pass"] is False
    slow = _document([0.2] * 30, [0.01] * 200, _cancel(seconds=1.5))
    assert slow["pass"] is False
    assert slow["targets"]["cancel_ack_s"]["numeric_pass"] is False  # type: ignore[index]
    assert slow["cancel_ack"]["owned_stop_pass"] is True  # type: ignore[index]
    assert slow["cancel_ack"]["chat_cancel"] is False  # type: ignore[index]
    assert slow["cancel_ack"]["method"] == lat.CANCEL_METHOD  # type: ignore[index]


def test_failure_text_is_scrubbed_in_the_document() -> None:
    doc = _document([], [], _cancel(seconds=None, stopped=False, cancelled=False, error=SECRET))
    blob = json.dumps(doc)
    assert SECRET not in blob
    assert "[REDACTED]" in blob
    cold = doc["cold_prompt"]
    assert isinstance(cold, dict)
    assert cold["n_ok"] == 0
    assert cold["p95_s"] is None


def test_parse_importtime_and_top_contributors() -> None:
    rows = lat.parse_importtime(IMPORT_TEXT)
    by_name = {row["module"]: row for row in rows}
    assert set(by_name) == {"alpha", "beta", "gamma"}
    assert by_name["beta"]["self_us"] == 80
    assert by_name["beta"]["cumulative_us"] == 500
    assert by_name["beta"]["occurrences"] == 2
    top = lat.top_import_contributors(rows, limit=2)
    assert [row["module"] for row in top] == ["gamma", "alpha"]
    many = [
        {"module": f"m{index:02d}", "self_us": index, "cumulative_us": index, "occurrences": 1}
        for index in range(25)
    ]
    assert len(lat.top_import_contributors(many)) == 20
    assert lat.top_import_contributors(many)[0]["module"] == "m24"
    tied = lat.top_import_contributors(
        [
            {"module": "b", "self_us": 5, "cumulative_us": 5, "occurrences": 1},
            {"module": "a", "self_us": 5, "cumulative_us": 9, "occurrences": 1},
            {"module": "c", "self_us": 5, "cumulative_us": 9, "occurrences": 1},
        ]
    )
    assert [row["module"] for row in tied] == ["a", "c", "b"]
    text = lat.format_import_profile(top, total=3)
    assert text.endswith("\n")
    assert "top 2 of 3" in text
    assert "gamma" in text


def test_child_env_strips_endpoint_secrets_and_pins_config() -> None:
    env = lat.child_env(
        {
            "PATH": "/usr/bin",
            "HOME": "/tmp/home",
            "MOONSHOT_API_KEY": SECRET,
            "FX1_BASE_URL": "https://example.invalid/v1",
            "MOONSHOT_API_URL": "https://example.invalid",
            "RANDOM_TOKEN": "drop-me",
            "PYTHONPATH": "/already",
        },
        "/cfg",
        "/sess",
    )
    assert "MOONSHOT_API_KEY" not in env
    assert "FX1_BASE_URL" not in env
    assert "MOONSHOT_API_URL" not in env
    assert "RANDOM_TOKEN" not in env
    assert SECRET not in "".join(env.values())
    assert env["FX1_CONFIG_DIR"] == "/cfg"
    assert env["FX1_SESSION_DIR"] == "/sess"
    assert env["NO_COLOR"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(lat._SRC)
    assert "/already" in parts


def _assert_prompt_matches_known(pattern: str) -> None:
    compiled = re.compile(pattern)
    matched = compiled.search("fx1-lite › ")
    assert matched is not None and matched.group(0) == "fx1-lite › "
    assert compiled.search("fx1 › ")
    assert compiled.search(lat.PROMPT)
    assert compiled.search("dipcatcher · fx1 @ unconfigured") is None


def test_chat_argv_omits_importtime_and_prompt_matches_console() -> None:
    from fx1.interactive.console import _LEGACY_PROMPT, _engine_prompt
    from fx1.interactive.profiles import MODELS

    argv = lat.chat_argv("/usr/bin/python3")
    assert argv == ["/usr/bin/python3", "-m", "dipcatcher_cli", "chat"]
    assert "-X" not in argv
    assert "importtime" not in argv
    compiled = re.compile(lat.prompt_pattern())
    assert compiled.search(_LEGACY_PROMPT)
    for model in MODELS:
        shown = _engine_prompt(SimpleNamespace(model=model))
        assert compiled.search(shown)
        assert shown.endswith(" › ")
    assert lat.prompt_pattern() == lat.prompt_pattern_from(lat.contract_markers())
    _assert_prompt_matches_known(lat.FALLBACK_PROMPT_PATTERN)


def test_windows_stop_uses_terminate_and_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lat.os, "name", "nt")
    proc = _Proc()
    lat._signal_proc(proc, signal.SIGTERM)  # type: ignore[arg-type]
    lat._signal_proc(proc, 9)  # type: ignore[arg-type]
    assert proc.terminated is True
    assert proc.killed is True
    assert proc.signaled == []


def test_posix_gap_records_failures_without_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lat, "pty_supported", lambda: False)
    cold = lat.measure_cold_runs(2, 1.0, "python", {})
    assert cold["samples_s"] == []
    assert cold["n_attempted"] == 2
    assert len(cold["failures"]) == 2
    assert cold["failures"][0]["error"] == lat.POSIX_PTY_ERROR
    events = lat.measure_input_events(3, 1.0, 1.0, "python", {})
    assert events["samples_s"] == []
    assert events["n_attempted"] == 3
    assert events["failures"][0]["error"] == lat.POSIX_PTY_ERROR


def test_import_retry_pauses_once_and_timeout_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake(argv: object, env: object, timeout_s: float) -> tuple[float | None, str | None]:
        del argv, env, timeout_s
        calls["n"] += 1
        if calls["n"] == 1:
            return None, "ImportError: dipcatcher_cli"
        return 0.2, None

    monkeypatch.setattr(lat, "pty_supported", lambda: True)
    monkeypatch.setattr(lat, "_time_to_prompt", fake)
    pauses: list[float] = []
    result = lat.measure_cold_runs(2, 5.0, sys.executable, {}, sleep=pauses.append)
    assert pauses == [30.0]
    assert result["samples_s"] == [0.2, 0.2]
    assert result["failures"] == []
    assert calls["n"] == 3

    monkeypatch.setattr(
        lat,
        "_time_to_prompt",
        lambda *args, **kwargs: (None, "timed out waiting for the chat prompt"),
    )
    quiet: list[float] = []
    failed = lat.measure_cold_runs(1, 1.0, "py", {}, sleep=quiet.append)
    assert quiet == []
    assert failed["samples_s"] == []
    assert "timed out" in failed["failures"][0]["error"]


def test_runner_loader_retries_import_and_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky() -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("transient")
        return lambda *args, **kwargs: None

    monkeypatch.setattr(lat, "_runner_from_procs", flaky)
    pauses: list[float] = []
    lat._load_runner(pauses.append)
    assert pauses == [30.0]
    assert calls["n"] == 2

    calls["n"] = 0

    def syntax() -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise SyntaxError("partial write")
        return lambda *args, **kwargs: None

    monkeypatch.setattr(lat, "_runner_from_procs", syntax)
    again: list[float] = []
    lat._load_runner(again.append)
    assert again == [30.0]


def test_cancel_runner_records_stop_without_chat_cancel() -> None:
    class Result:
        cancelled = True
        exit_code = 130
        spawn_error = None

    def runner(
        argv: list[str],
        *,
        cancel_event: object,
        env: object,
        emit: object,
        timeout_s: float,
    ) -> Result:
        del argv, env, timeout_s
        emit("proc.spawn", {"pid": 1})  # type: ignore[operator]
        assert cancel_event.wait(2)  # type: ignore[attr-defined]
        return Result()

    result = lat.measure_cancel("py", {"PATH": "/bin"}, timeout_s=2, runner=runner)
    assert result["stopped"] is True
    assert result["cancelled"] is True
    assert result["chat_cancel"] is False
    assert result["method"] == lat.CANCEL_METHOD
    assert result["exit_code"] == 130
    assert result["seconds"] >= 0
    assert "run_owned" in result["reason"]


def test_cancel_hides_exception_text_and_reports_a_missed_spawn() -> None:
    def explode(
        argv: list[str],
        *,
        cancel_event: object,
        env: object,
        emit: object,
        timeout_s: float,
    ) -> None:
        del argv, cancel_event, env, timeout_s
        emit("proc.spawn", {})  # type: ignore[operator]
        raise RuntimeError(SECRET)

    failed = lat.measure_cancel("py", {}, timeout_s=2, runner=explode)
    assert failed["stopped"] is False
    assert SECRET not in str(failed["error"])
    assert failed["error"] == "RuntimeError"

    def silent(
        argv: list[str],
        *,
        cancel_event: object,
        env: object,
        emit: object,
        timeout_s: float,
    ) -> None:
        del argv, cancel_event, env, emit, timeout_s
        return None

    missed = lat.measure_cancel("py", {}, timeout_s=0.05, runner=silent)
    assert missed["stopped"] is False
    assert "spawn" in str(missed["error"])


def test_cli_defaults_and_atomic_0600(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, object] = {}

    def cold(request: lat.MeasureRequest, env: object) -> dict[str, object]:
        del env
        seen["runs"] = request.runs
        seen["reference"] = request.reference_hardware
        return _cold([0.2] * request.runs)

    def events(request: lat.MeasureRequest, env: object) -> dict[str, object]:
        del env
        seen["events"] = request.events
        return _events([0.01] * request.events)

    def cancel(request: lat.MeasureRequest, env: object) -> dict[str, object]:
        del request, env
        return _cancel()

    out = tmp_path / "nested" / "latency.json"
    rc = lat.main(
        ["--out", str(out)],
        cold_fn=cold,
        event_fn=events,
        cancel_fn=cancel,
    )
    assert rc == 0
    assert seen == {"runs": 30, "events": 200, "reference": False}
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["pass"] is True
    assert document["p5_gate_closed"] is False
    assert document["cold_prompt"]["command"] == ["python", "-m", "dipcatcher_cli", "chat"]
    assert "-X" not in document["cold_prompt"]["command"]
    if os.name == "posix":
        assert out.stat().st_mode & 0o777 == 0o600
        assert out.parent.stat().st_mode & 0o777 == 0o700
    assert list(out.parent.glob(".latency.json.*.tmp")) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pass=true" in captured.err
    assert SECRET not in captured.err


def test_cli_small_sample_fails_closed_and_stdout_is_one_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def cold(request: lat.MeasureRequest, env: object) -> dict[str, object]:
        del env
        assert request.runs == 3
        return _cold([0.2, 0.1, 0.3])

    def events(request: lat.MeasureRequest, env: object) -> dict[str, object]:
        del env
        assert request.events == 5
        return _events([0.01] * 5)

    out = tmp_path / "small.json"
    rc = lat.main(
        ["--runs", "3", "--events", "5", "--out", str(out), "--reference-hardware"],
        cold_fn=cold,
        event_fn=events,
        cancel_fn=lambda request, env: _cancel(),
    )
    assert rc == 1
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["reference_hardware"] is True
    assert document["p5_gate_closed"] is False
    assert document["pass"] is False
    assert document["targets"]["cold_prompt_p95_s"]["numeric_pass"] is True
    assert document["targets"]["cold_prompt_p95_s"]["sample_pass"] is False
    assert document["cold_prompt"]["n_ok"] == 3
    assert document["input_ack"]["n_ok"] == 5

    stdout_rc = lat.main(
        ["--runs", "3", "--events", "5"],
        cold_fn=cold,
        event_fn=events,
        cancel_fn=lambda request, env: _cancel(),
    )
    assert stdout_rc == 1
    captured = capsys.readouterr()
    loaded = json.loads(captured.out)
    assert loaded["schema"] == lat.SCHEMA
    assert loaded["cold_prompt"]["n_ok"] == 3
    assert "pass=false" in captured.err


def test_usage_errors_and_help() -> None:
    assert lat.main(["--runs", "0"]) == 2
    assert lat.main(["--events", "-3"]) == 2
    assert lat.main(["--runs", "1001"]) == 2
    assert lat.main(["--cold-timeout-s", "0"]) == 2
    assert lat.main(["--event-timeout-s", "nan"]) == 2
    assert lat.main(["--cancel-timeout-s", "121"]) == 2
    assert lat.main(["--not-a-flag"]) == 2
    assert lat.main(["--help"]) == 0


def test_import_profile_mode_does_not_measure(capsys: pytest.CaptureFixture[str]) -> None:
    called = {"n": 0}

    def cold(request: lat.MeasureRequest, env: object) -> dict[str, object]:
        del request, env
        called["n"] += 1
        return _cold([0.2] * 30)

    rc = lat.main(
        ["--import-profile", "--runs", "3"],
        cold_fn=cold,
        import_fn=lambda request, env: IMPORT_TEXT,
    )
    assert rc == 0
    assert called["n"] == 0
    out = capsys.readouterr().out
    assert "gamma" in out
    assert out.strip().splitlines()[2].split()[-1] == "gamma"
    empty = lat.main(["--import-profile"], import_fn=lambda request, env: "noise\n")
    assert empty == 1
    assert "no import time rows" in capsys.readouterr().err


def test_sleeper_stops_when_terminated() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(lat.SCRIPT_PATH), "--sleeper"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], 2)
    assert ready
    assert proc.stdout.readline() == b"ready\n"
    proc.terminate()
    assert proc.wait(timeout=2) is not None


@pytest.mark.skipif(os.name != "posix", reason="PTY measurement requires POSIX")
def test_real_pty_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("MOONSHOT_API_KEY", SECRET)
    out = tmp_path / "latency.json"
    rc = lat.main(
        [
            "--runs",
            "1",
            "--events",
            "3",
            "--out",
            str(out),
            "--cold-timeout-s",
            "15",
            "--event-timeout-s",
            "5",
            "--cancel-timeout-s",
            "10",
        ]
    )
    text = out.read_text(encoding="utf-8")
    assert SECRET not in text
    document = json.loads(text)
    assert document["schema"] == lat.SCHEMA
    assert document["reference_hardware"] is False
    assert document["p5_gate_closed"] is False
    assert document["percentile_method"] == "nearest_rank"
    assert rc == 1
    cold = document["cold_prompt"]
    assert cold["failures"] == []
    assert cold["n_ok"] == 1
    from fx1.interactive.console import _LEGACY_PROMPT, _engine_prompt
    from fx1.interactive.profiles import DEFAULT_MODEL

    compiled = re.compile(cold["prompt_pattern"])
    assert compiled.search(_LEGACY_PROMPT)
    assert compiled.search(_engine_prompt(SimpleNamespace(model=DEFAULT_MODEL)))
    assert document["prompt_pattern"] == cold["prompt_pattern"]
    assert cold["max_s"] < 10
    assert "-X" not in cold["command"]
    assert "importtime" not in cold["command"]
    ack = document["input_ack"]
    assert ack["n_ok"] == 3
    assert ack["failures"] == []
    assert ack["max_s"] < 10
    assert ack["command_cycle"] == list(lat.LOCAL_COMMANDS)
    cancel = document["cancel_ack"]
    assert cancel["method"] == lat.CANCEL_METHOD
    assert cancel["chat_cancel"] is False
    assert cancel["stopped"] is True
    assert cancel["seconds"] < 10
    assert document["targets"]["cold_prompt_p95_s"]["sample_pass"] is False
    assert document["targets"]["input_ack_p95_s"]["sample_pass"] is False
    assert document["targets"]["cold_prompt_p95_s"]["required_samples"] == 30
    assert document["environment"]["cpu_count"] == os.cpu_count()
    assert document["environment"]["python"] == platform.python_version()
    assert isinstance(document["environment"]["load_average"], list)
    assert len(document["environment"]["load_average"]) == 3
    assert "pythonhashseed" in document["environment"]
    assert out.stat().st_mode & 0o777 == 0o600
    profile = lat.main(["--import-profile", "--cold-timeout-s", "15"])
    captured = capsys.readouterr()
    assert profile == 0
    assert "self_us" in captured.out
    assert "fx1" in captured.out or "dipcatcher_cli" in captured.out
    assert SECRET not in captured.out
    assert SECRET not in captured.err
