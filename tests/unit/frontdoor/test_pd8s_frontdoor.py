"""PD-8S FrontDoor acceptance: credentials, paths, honesty, superpower exit."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dipcatcher_cli.paths import repo_root, resolve_repo_path
from dipcatcher_cli.truthfulness import (
    env_documentation_surfaces,
    env_example_variables,
    receipt_census,
)
from fx1.harness import Harness
from fx1.honesty import validate_fx1_output
from fx1.interactive import profiles
from fx1.interactive.actions import harness_run
from fx1.interactive.concierge import Concierge
from fx1.interactive.profiles import ENV_API_KEY
from quant_fund.cli.main import app as lab_app
from quant_fund.pipeline.doctor import doctor as lab_doctor

SECRET = "session-only-secret-key-0123456789"


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return tmp_path


def test_use_does_not_leak_api_key_to_lab_subprocess(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles.set_endpoint("fx1", SECRET, "https://api.example.test/v1")
    concierge = Concierge(model="fx1", journal=None)
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


def test_harness_subprocess_env_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_KEY, SECRET)
    popens: list[dict[str, object]] = []

    class _Proc:
        returncode = 0

        def poll(self) -> int:
            return 0

        @property
        def stdout(self) -> None:
            return None

        @property
        def stderr(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> _Proc:
        popens.append(kwargs)
        return _Proc()

    monkeypatch.setattr("fx1.harness.subprocess.Popen", fake_popen)
    monkeypatch.setattr("fx1.harness.shutil.which", lambda _: "/usr/bin/dipcatcher")
    Harness().run("doctor")
    assert popens
    env = popens[0].get("env")
    assert isinstance(env, dict)
    assert env.get(ENV_API_KEY) is None


def test_repo_root_doctor_from_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir("/tmp")
    cfg = tmp_path / "research.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    result = CliRunner().invoke(lab_app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "core_imports" in result.stdout
    assert "Traceback" not in result.stdout


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


def test_readme_receipt_census_matches_git() -> None:
    census = receipt_census()
    assert census["json_total"] == 404
    assert census["json_top_level"] == 397
    assert census["legacy_unsealed_json"] == 7
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    assert "404" in readme and "397" in readme


def test_env_example_vars_documented_in_operator_surfaces() -> None:
    declared = env_example_variables()
    documented = env_documentation_surfaces()
    undocumented = sorted(declared - documented)
    assert not undocumented, f".env.example vars missing from README/AGENTS: {undocumented[:10]}"


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
