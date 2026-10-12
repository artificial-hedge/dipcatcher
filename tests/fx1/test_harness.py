"""Harness tests: registry fail-closed, runner injection, config containment."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from fx1.harness import Harness, HarnessResult, HarnessRole
from fx1.interactive.procs import ProcResult


def _fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, f"ran: {' '.join(argv)}", ""


def test_registry_covers_all_four_roles():
    harness = Harness(runner=_fake_runner)
    roles = {c.role for c in harness.list_commands()}
    assert roles == {
        HarnessRole.DATA_ENGINE,
        HarnessRole.EVALUATION,
        HarnessRole.VERIFICATION,
        HarnessRole.MODEL_TRAINING,
    }


def test_registry_covers_full_lab_surface():
    names = {c.name for c in Harness(runner=_fake_runner).list_commands()}
    expected = {
        "doctor",
        "verify-research",
        "validate",
        "monitor",
        "ingest",
        "collect",
        "build-features",
        "build-labels",
        "research",
        "northset",
        "backtest",
        "forecast",
        "kronos-forecast",
        "candle-book",
        "kyle-ofi",
        "session-book",
        "vendor-book-map",
        "book-panel",
        "report",
        "tearsheet",
        "train",
        "optimize",
        "paper",
    }
    assert expected <= names
    # Hidden/network-serving surfaces are deliberately unreachable by fx-1.
    assert "lab" not in names and "api" not in names


def test_role_filter():
    harness = Harness(runner=_fake_runner)
    verification = harness.list_commands(role=HarnessRole.VERIFICATION)
    assert verification and all(c.role == HarnessRole.VERIFICATION for c in verification)


def test_unregistered_command_fails_closed():
    harness = Harness(runner=_fake_runner)
    with pytest.raises(KeyError, match="not a registered harness command"):
        harness.run("live-trading")


def test_run_dispatches_through_injected_runner():
    harness = Harness(runner=_fake_runner)
    result = harness.run("verify-research")
    assert result.ok
    assert "verify-research" in result.stdout


def test_config_contained_to_configs_dir(tmp_path: Path):
    harness = Harness(runner=_fake_runner)
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("research", config=tmp_path / "evil.yaml")


def test_default_runner_targets_dipcatcher_cli():
    harness = Harness()
    # Registry argv always targets the dipcatcher harness CLI surface.
    assert all(isinstance(c.argv, list) and c.argv for c in harness.list_commands())


def test_harness_result_new_fields_default() -> None:
    result = HarnessResult(command="doctor", exit_code=0, stdout="a", stderr="b")
    assert result.ok
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.truncated is False
    assert result.duration_s == 0.0


def test_legacy_runner_receives_exactly_two_arguments() -> None:
    seen: dict[str, object] = {}

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen["argv"] = list(argv)
        seen["timeout"] = timeout_s
        return 7, "out", "err"

    result = Harness(runner=_runner).run(
        "doctor",
        ["--help"],
        cancel_event=threading.Event(),
        env_extra={"A": "b"},
        step="s",
    )
    assert result.exit_code == 7
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.ok is False
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.truncated is False
    assert result.duration_s == 0.0
    assert seen["argv"] == ["doctor", "--help"]
    assert seen["timeout"] == 300


def test_legacy_runner_without_extended_flag_ignores_new_kwargs() -> None:
    class _Legacy:
        supports_extended = False

        def __call__(self, argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            return 0, "plain", ""

    result = Harness(runner=_Legacy()).run("doctor", cancel_event=threading.Event(), step="s")
    assert result.ok
    assert result.stdout == "plain"
    assert result.timed_out is False


def test_legacy_timeout_expired_becomes_result() -> None:
    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        raise subprocess.TimeoutExpired(argv, timeout_s, output=b"partial", stderr=b"nope")

    result = Harness(runner=_runner).run("monitor")
    assert result.exit_code == 124
    assert result.timed_out is True
    assert result.ok is False
    assert result.stdout == "partial"
    assert "nope" in result.stderr
    assert "[timeout]" in result.stderr


def test_extended_runner_receives_keywords_and_maps_result() -> None:
    class _Recorder:
        supports_extended = True

        def __call__(
            self,
            argv: list[str],
            timeout_s: int,
            *,
            cancel_event: threading.Event | None = None,
            env_extra: dict[str, str] | None = None,
            step: str | None = None,
        ) -> ProcResult:
            self.got = (list(argv), timeout_s, env_extra, step, cancel_event)
            return ProcResult(
                argv_summary=list(argv),
                exit_code=124,
                stdout="o",
                stderr="e",
                timed_out=True,
                cancelled=False,
                truncated=True,
                spawn_error=None,
                duration_s=1.5,
                pid=4,
                pgid=4,
            )

    recorder = _Recorder()
    cancel = threading.Event()
    result = Harness(runner=recorder).run(
        "paper",
        cancel_event=cancel,
        env_extra={"Q": "1"},
        step="st",
    )
    assert result.exit_code == 124
    assert result.timed_out is True
    assert result.truncated is True
    assert result.duration_s == 1.5
    assert result.stdout == "o"
    assert recorder.got[0] == ["paper"]
    assert recorder.got[1] == 1800
    assert recorder.got[2] == {"Q": "1"}
    assert recorder.got[3] == "st"
    assert recorder.got[4] is cancel


def test_extra_config_args_stay_inside_configs() -> None:
    harness = Harness(runner=_fake_runner)
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config", "/tmp/evil.yaml"])
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config=/tmp/evil.yaml"])


def test_config_inside_configs_reaches_argv() -> None:
    seen: dict[str, list[str]] = {}

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen["argv"] = list(argv)
        return 0, "", ""

    cfg = Path("configs/research.yaml")
    result = Harness(runner=_runner).run("doctor", config=cfg)
    assert result.ok
    assert seen["argv"][0] == "doctor"
    assert "--config" in seen["argv"]
    assert str(cfg.resolve()) in seen["argv"]


def test_importing_harness_does_not_import_procs() -> None:
    script = (
        "import sys\n"
        "import fx1.harness\n"
        "assert 'fx1.interactive.concierge' not in sys.modules\n"
        "assert 'fx1.interactive.procs' not in sys.modules\n"
        "harness = fx1.harness.Harness()\n"
        "assert harness.list_commands()\n"
        "assert harness._runner.supports_extended is True\n"
        "assert 'fx1.interactive.procs' not in sys.modules\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/workspace",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_owned_runner_argv_and_scrub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = "sk-owned-runner-secret"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    captured: dict[str, object] = {}

    def _fake_run(argv: list[str], limits: object, **kwargs: object) -> ProcResult:
        captured["argv"] = list(argv)
        captured["env"] = dict(kwargs["env"])  # type: ignore[arg-type]
        captured["step"] = kwargs.get("step")
        captured["cancel"] = kwargs.get("cancel_event")
        return ProcResult(
            argv_summary=["x"],
            exit_code=0,
            stdout="ok",
            stderr="",
            timed_out=False,
            cancelled=False,
            truncated=False,
            spawn_error=None,
            duration_s=0.01,
            pid=1,
            pgid=1,
        )

    import fx1.interactive.procs as procs

    monkeypatch.setattr(procs, "run_owned", _fake_run)
    cancel = threading.Event()
    result = Harness().run(
        "doctor",
        cancel_event=cancel,
        env_extra={"QUANT_MODE": "lab"},
        step="step-a",
    )
    assert result.ok
    assert result.stdout == "ok"
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[:3] == [sys.executable, "-m", "dipcatcher_cli"]
    assert argv[3:] == ["doctor"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert "MOONSHOT_API_KEY" not in env
    assert secret not in env.values()
    assert env["DIPCATCHER_CHILD"] == "1"
    assert env["FX1_CONFIG_DIR"] == str(tmp_path)
    assert env["QUANT_MODE"] == "lab"
    assert captured["step"] == "step-a"
    assert captured["cancel"] is cancel


def test_owned_runner_doctor_help(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = "sk-doctor-help-secret"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FX1_BASE_URL", "https://user:secret@example.invalid/v1")
    result = Harness().run("doctor", extra_args=["--help"])
    assert result.exit_code == 0, result.stderr
    assert "doctor" in result.stdout.lower()
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.duration_s > 0
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "example.invalid" not in result.stdout
