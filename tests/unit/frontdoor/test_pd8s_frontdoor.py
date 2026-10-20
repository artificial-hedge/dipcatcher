"""PD-8S FrontDoor acceptance: credentials, paths, honesty, superpower exit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dipcatcher_cli.paths import repo_root, resolve_repo_path
from dipcatcher_cli.truthfulness import (
    RUNTIME_ENV_ALLOWLIST,
    env_documentation_surfaces,
    env_example_all_variables,
    env_example_variables,
    receipt_census,
    source_environ_names,
)
from fx1.harness import Harness, _run_subprocess
from fx1.honesty import validate_fx1_output
from fx1.interactive import profiles
from fx1.interactive.actions import harness_run, run_tool
from fx1.interactive.concierge import Concierge
from fx1.interactive.profiles import ENV_API_KEY, scrub_credentials
from fx1.webresearch.engine import Researcher
from quant_fund.cli.main import app as lab_app
from quant_fund.pipeline.doctor import doctor as lab_doctor
from quant_fund.utils.lab_paths import discover_repo_root

SECRET = "session-only-secret-key-0123456789"


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv("FX1_WEB_RESEARCH", raising=False)
    return tmp_path


def test_use_does_not_leak_api_key_to_lab_subprocess(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles.set_endpoint("fx1", SECRET, "https://api.example.test/v1")
    concierge = Concierge(model="fx1", journal=None, persist=False)
    concierge.handle_line("/use fx1")
    assert os.environ.get(ENV_API_KEY) != SECRET

    captured: dict[str, str] = {}

    def fake_run(self, name: str, extra_args: list[str] | None = None, **kwargs: object) -> object:
        from fx1.harness import HarnessResult

        captured.update(os.environ)
        return HarnessResult(command=name, exit_code=0, stdout="{}", stderr="")

    monkeypatch.setattr(Harness, "run", fake_run)
    assert harness_run("doctor", model="fx1") == 0
    assert captured.get(ENV_API_KEY) is None


def test_use_then_spawned_lab_child_has_no_api_key(isolated_store: Path) -> None:
    """Acceptance: a real subprocess after /use does not inherit the model key."""
    profiles.set_endpoint("fx1", SECRET, "https://api.example.test/v1")
    concierge = Concierge(model="fx1", journal=None, persist=False)
    concierge.handle_line("/use fx1")
    assert os.environ.get(ENV_API_KEY) != SECRET
    env = profiles.subprocess_env_for_model("fx1", inject_credentials=False)
    assert env.get(ENV_API_KEY) is None
    script = "import os,sys; sys.stdout.write(os.environ.get('MOONSHOT_API_KEY') or '')"
    outcome = _run_subprocess(
        sys.executable,
        ["-c", script],
        timeout_s=15,
        cancel_event=None,
        grace_s=1.0,
        poll_s=0.05,
        max_output_chars=4_000,
        env=env,
        cwd=str(discover_repo_root()),
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert SECRET not in outcome.stdout
    assert outcome.stdout == ""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        env=scrub_credentials(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == ""


def test_run_tool_lab_binary_scrubs_credentials(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles.set_endpoint("fx1", SECRET, "https://api.example.test/v1")
    calls: list[dict[str, object]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append({"argv": list(argv), **kwargs})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("fx1.interactive.actions.subprocess.run", fake_run)
    monkeypatch.setattr("fx1.interactive.actions.shutil.which", lambda name: f"/usr/bin/{name}")
    assert run_tool("dipcatcher", ["doctor"], "fx1") == 0
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env.get(ENV_API_KEY) is None


def test_harness_subprocess_env_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.harness import RunnerOutcome

    monkeypatch.setenv(ENV_API_KEY, SECRET)
    captured: dict[str, object] = {}

    def fake_run_subprocess(
        executable: str,
        argv: list[str],
        timeout_s: int,
        cancel_event: object,
        *,
        grace_s: float,
        poll_s: float,
        max_output_chars: int,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> RunnerOutcome:
        del executable, argv, timeout_s, cancel_event, grace_s, poll_s, max_output_chars
        captured["env"] = env
        captured["cwd"] = cwd
        return RunnerOutcome(exit_code=0, stdout="{}", stderr="")

    monkeypatch.setattr("fx1.harness.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("fx1.harness._run_subprocess", fake_run_subprocess)
    result = Harness().run("doctor")
    assert result.exit_code == 0
    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get(ENV_API_KEY) is None
    assert SECRET not in env.values()
    cwd = captured.get("cwd")
    assert isinstance(cwd, str) and cwd
    assert Path(cwd) == discover_repo_root()


def test_repo_root_doctor_from_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir("/tmp")
    monkeypatch.setenv("DIPCATCHER_ROOT", str(repo_root()))
    cfg = tmp_path / "research.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    result = CliRunner().invoke(lab_app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "core_imports" in result.stdout
    assert "Traceback" not in result.stdout
    assert "dir_raw:" in result.stdout


def test_doctor_default_config_from_tmp_no_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir("/tmp")
    monkeypatch.setenv("DIPCATCHER_ROOT", str(repo_root()))
    result = CliRunner().invoke(lab_app, ["doctor"])
    assert "Traceback" not in (result.stdout + (result.stderr or ""))
    assert "core_imports" in result.stdout


def test_resolve_repo_path_finds_configs() -> None:
    path = resolve_repo_path("configs/research.yaml")
    assert path.is_file()
    assert path.is_relative_to(repo_root())


def test_sim_live_default_stdout_passes_honesty_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.cli.test_cli_main_cov import _data_config, _patch_sim_live

    cfg = _data_config(tmp_path)
    _patch_sim_live(monkeypatch)
    result = CliRunner().invoke(lab_app, ["sim-live", "--config", str(cfg), "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    validate_fx1_output(result.output)
    assert "sharpe=" not in result.output.lower()


def test_sim_live_diagnostic_includes_sharpe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.cli.test_cli_main_cov import _data_config, _patch_sim_live

    cfg = _data_config(tmp_path)
    _patch_sim_live(monkeypatch)
    result = CliRunner().invoke(
        lab_app,
        ["sim-live", "--config", str(cfg), "--out", str(tmp_path), "--diagnostic"],
    )
    assert result.exit_code == 0, result.output
    assert "sharpe=" in result.output.lower()
    assert "DIAGNOSTIC" in result.output


def test_superpower_sync_partial_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.interactive import app as fxi_app
    from fx1.interactive.superpower import CapabilityRun, SuperpowerPlan, SuperpowerResult

    plan = SuperpowerPlan(goal="probe", capabilities=[], source="test")
    partial = SuperpowerResult(
        goal="probe",
        plan=plan,
        summary="partial run",
        runs=[CapabilityRun("doctor", "doctor", "failed", "synthetic failure")],
    )
    assert partial.partial is True

    class _Op:
        def __init__(self) -> None:
            self.job = SimpleNamespace(error=None, result=partial)

        def start_superpower(
            self, goal: str, *, sync: bool = False, allow_web: bool = False
        ) -> bool:
            return False

        def join_job(self, timeout_s: float | None = None) -> None:
            return None

        def close(self, reason: str = "") -> None:
            return None

        run_budget = SimpleNamespace(max_wall_s=60)

    monkeypatch.setattr("fx1.interactive.app.concierge.Concierge", lambda **k: _Op())
    result = CliRunner().invoke(fxi_app.app, ["superpower", "--sync", "probe goal"])
    assert result.exit_code == 1


def test_superpower_without_web_opt_in_makes_no_network_call(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    searches: list[str] = []

    class _Searcher:
        def search(self, query: str, max_results: int = 8) -> list[object]:
            searches.append(query)
            raise AssertionError(f"network search invoked for {query!r}")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("urllib requested without web opt-in")

    monkeypatch.delenv("FX1_WEB_RESEARCH", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", boom)
    concierge = Concierge(
        model="fx1",
        researcher=Researcher(searcher=_Searcher()),  # type: ignore[arg-type]
        persist=False,
        journal=None,
        harness=Harness(runner=lambda argv, timeout_s: (0, "{}", "")),
    )
    concierge.start_superpower("what is the latest news on the Fed rate decision", sync=True)
    assert searches == []
    job = concierge.job
    assert job is not None
    result = job.result
    assert result is not None
    assert "web_research" not in result.plan.ids()


def test_readme_receipt_census_matches_git() -> None:
    census = receipt_census()
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    assert str(census["json_total"]) in readme
    assert str(census["json_top_level"]) in readme
    assert census["legacy_unsealed_json"] == 7

def test_env_example_vars_documented_in_operator_surfaces() -> None:
    declared = env_example_variables()
    documented = env_documentation_surfaces()
    undocumented = sorted(declared - documented)
    assert not undocumented, f".env.example vars missing from README/AGENTS: {undocumented[:10]}"


def test_superpower_web_search_requires_explicit_opt_in() -> None:
    from unittest.mock import MagicMock

    from fx1.interactive.superpower import (
        SuperpowerContext,
        SuperpowerPlan,
        SuperpowerResult,
        _stage_web_research,
    )

    plan = SuperpowerPlan(goal="public macro question", capabilities=[], source="test")
    ctx = SuperpowerContext(allow_web=False)
    result = SuperpowerResult(goal=plan.goal, plan=plan, summary="")
    researcher = MagicMock()
    _stage_web_research(plan, ctx, researcher, result, [])
    researcher.run.assert_not_called()
    web = next(run for run in result.runs if run.id == "web_research")
    assert web.status == "skipped"
    assert "opt-in" in web.detail.lower()


def test_superpower_capabilities_from_tmp_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.interactive.superpower import describe_capabilities

    monkeypatch.chdir("/tmp")
    text = describe_capabilities()
    assert "web_research" in text
    assert "Traceback" not in text


def test_env_example_covers_source_environ_reads() -> None:
    missing = sorted(source_environ_names() - env_example_all_variables() - RUNTIME_ENV_ALLOWLIST)
    assert not missing, f"src os.environ/getenv keys missing from .env.example: {missing[:20]}"


def test_lab_doctor_reports_created_dirs_separately(tmp_path: Path) -> None:
    cfg = tmp_path / "research.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    info = lab_doctor(str(cfg))
    for part in ("raw", "bronze", "silver", "gold", "metadata"):
        assert info[f"dir_{part}"] == "created"
        assert info[f"dir_{part}_created"] == "yes"
    info2 = lab_doctor(str(cfg))
    for part in ("raw", "bronze", "silver", "gold", "metadata"):
        assert info2[f"dir_{part}"] == "ok"
        assert info2[f"dir_{part}_created"] == "no"
