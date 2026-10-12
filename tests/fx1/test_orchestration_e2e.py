"""Vertical slice: real SYNTHETIC vol-bench, sealed receipt, grounded answer."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from fx1.harness import Harness
from fx1.honesty import validate_fx1_output
from fx1.interactive.contracts import CapabilityCatalog
from fx1.interactive.evidence_ledger import EvidenceLedger, Verdict
from fx1.interactive.families import FamilyClient, harness_family_runner
from fx1.interactive.jobs import JobLedger, JobState
from fx1.interactive.orchestration import load_catalog
from fx1.interactive.run_report import compose_run_answer
from fx1.interactive.runctx import new_run_id, new_session_id
from fx1.interactive.superpower import SuperpowerContext, SuperpowerPlan, plan_goal, run_plan

_FORBIDDEN = ("sharpe", "sortino", "calmar", "p&l", "pnl")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.chdir(tmp_path)


def _vol_document() -> dict[str, object]:
    """Schema-valid vol-bench slice using the overlay's tiny bounds."""
    return {
        "schema": "dipcatcher.capabilities.v1",
        "generated_by": {
            "n_lab_commands": 1,
            "overlay_sha256": "ab" * 32,
        },
        "capabilities": [
            {
                "id": "vol-bench",
                "version": "1",
                "purpose": "Score SYNTHETIC volatility forecasts with QLIKE and MSE.",
                "tasks": ["qlike_scoring", "volatility_forecast_evaluation"],
                "aliases": ["realized variance", "vol bench"],
                "command": {
                    "argv": ["vol-bench"],
                    "role": "evaluation",
                    "read_only": False,
                    "writes": ["receipts"],
                    "consequential": False,
                    "timeout_s": 900,
                },
                "args": [
                    {
                        "name": "--models",
                        "kind": "option",
                        "type": "csv",
                        "required": False,
                        "default": "rv_roll",
                        "repeatable": False,
                        "item_type": "str",
                        "help": "Vol models.",
                    },
                    {
                        "name": "--shards",
                        "kind": "option",
                        "type": "csv",
                        "required": False,
                        "default": "garch_vol",
                        "repeatable": False,
                        "item_type": "str",
                        "help": "Shards.",
                    },
                    {
                        "name": "--horizons",
                        "kind": "option",
                        "type": "csv",
                        "required": False,
                        "default": "1",
                        "repeatable": False,
                        "item_type": "int",
                        "help": "Horizons.",
                    },
                    {
                        "name": "--min-history",
                        "kind": "option",
                        "type": "int",
                        "required": False,
                        "default": 64,
                        "repeatable": False,
                        "min": 32,
                        "max": 400,
                        "help": "Fit bars.",
                    },
                    {
                        "name": "--n-origins",
                        "kind": "option",
                        "type": "int",
                        "required": False,
                        "default": 10,
                        "repeatable": False,
                        "min": 10,
                        "max": 24,
                        "help": "Origins.",
                    },
                    {
                        "name": "--out-dir",
                        "kind": "option",
                        "type": "path",
                        "required": False,
                        "repeatable": False,
                        "path_role": "output_dir",
                        "help": "Receipt directory.",
                    },
                    {
                        "name": "--receipt-version",
                        "kind": "option",
                        "type": "int",
                        "required": False,
                        "default": 2,
                        "repeatable": False,
                        "min": 2,
                        "max": 2,
                        "help": "Receipt schema version.",
                    },
                ],
                "inputs": {
                    "data_labels": ["SYNTHETIC"],
                    "requires_dataset": False,
                    "availability_timing": "n/a — seeded synthetic shards",
                },
                "outputs": {
                    "scores": ["QLIKE", "MSE"],
                    "receipt_schema": "receipt.v2",
                    "receipt_flag": ["--receipt-version", "2"],
                    "receipt_stdout_marker": "receipt=",
                },
                "dependencies": [],
                "post_checks": [{"kind": "verify_receipt", "from": "receipt_stdout_marker"}],
                "limits": {"max_runtime_s": 900, "max_output_chars": 12000},
                "qualification": {
                    "status": "qualified",
                    "evidence": ["tests/unit/research/test_vol_bench.py"],
                    "ruleset": "overlay-v1",
                    "audited_on": "2026-10-12",
                },
                "evidence_class": "SYNTHETIC",
                "executable": True,
                "executable_reason": "",
            }
        ],
    }


def _copy_config(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    destination = tmp_path / "configs"
    destination.mkdir()
    shutil.copy(repo / "configs" / "research.yaml", destination / "research.yaml")
    shutil.copy(repo / "configs" / "base.yaml", destination / "base.yaml")


def _receipt_names(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.name for path in root.glob("vol_bench_*.json")}


def test_real_vol_bench_receipt_is_verified_and_tamper_downgrades(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    before = _receipt_names(repo / "receipts")
    _copy_config(tmp_path)
    run_dir = tmp_path / "run"
    catalog = CapabilityCatalog.from_json(_vol_document())
    plan = plan_goal(
        "score realized variance",
        catalog=catalog,
        roots={"output_dir": [run_dir]},
    )
    assert [step.capability_id for step in plan.typed_steps] == ["vol-bench"]

    def model(prompt: str) -> str:
        match = re.search(r"receipt=(\S+)", prompt)
        assert match is not None
        return f"SYNTHETIC QLIKE volatility result is a verified receipt {match.group(1)}"

    ledger = JobLedger(tmp_path / "jobs")
    result = run_plan(
        plan,
        SuperpowerContext(
            model_fn=model,
            harness=Harness(),
            catalog=catalog,
            run_dir=run_dir,
            roots={"output_dir": [run_dir]},
            job_ledger=ledger,
            session_id=new_session_id(),
            run_id=new_run_id(),
        ),
    )
    evidence = EvidenceLedger.from_dict(result.evidence)
    verified = evidence.verified_receipts()
    assert len(verified) == 1
    assert verified[0].verdict is Verdict.VERIFIED
    assert verified[0].data_label == "SYNTHETIC"
    assert result.status == "verified"
    assert "SYNTHETIC" in result.summary
    assert "verified receipt" in result.summary
    assert result.summary.count("Evidence (harness-verified") == 1
    assert validate_fx1_output(result.summary) == result.summary
    lowered = result.summary.lower()
    for token in _FORBIDDEN:
        assert token not in lowered
    assert " nav" not in lowered

    job = ledger.get(result.job_id or "")
    assert job.state is JobState.SUCCEEDED
    argv = job.steps[0].argv_summary
    assert "--out-dir" in argv
    out_dir = Path(argv[argv.index("--out-dir") + 1])
    assert run_dir in out_dir.parents or out_dir == run_dir
    assert not any(
        token == "receipts" or token.rstrip("/\\").endswith("receipts") for token in argv
    )
    receipt = next(run_dir.rglob("vol_bench_*.json"))
    assert str(tmp_path) in str(receipt)
    assert _receipt_names(repo / "receipts") == before

    raw = bytearray(receipt.read_bytes())
    flipped = _flip_digit(raw)
    receipt.write_bytes(flipped)
    tampered = EvidenceLedger()
    item = tampered.add_receipt(receipt, roots=[run_dir])
    assert item.verdict is Verdict.INVALID
    downgraded = compose_run_answer(
        f"SYNTHETIC QLIKE volatility result is a verified receipt {receipt}",
        tampered,
        verified_paths=set(),
        artifact_paths={str(receipt), item.ref, receipt.name},
    )
    assert downgraded.status != "verified"
    assert "unverified and partial" in downgraded.summary
    assert "SYNTHETIC" in downgraded.summary
    assert validate_fx1_output(downgraded.summary) == downgraded.summary
    assert downgraded.summary.count("Evidence (harness-verified") == 1


def _flip_digit(raw: bytearray) -> bytes:
    """Change one decimal digit so the JSON still parses and the seal breaks."""
    body = json.loads(bytes(raw))
    _bump_number(body)
    return (json.dumps(body) + "\n").encode("utf-8")


def _bump_number(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                value[key] = item + 1 if isinstance(item, int) else item + 1.0
                return True
            if _bump_number(item):
                return True
        return False
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                value[index] = item + 1 if isinstance(item, int) else item + 1.0
                return True
            if _bump_number(item):
                return True
    return False


# ---------------------------------------------------------------------------
# Real catalog (live ``capabilities list --json``) through the real planner,
# the real harness, and the real verifier.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]


def _tree_snapshot(*roots: Path) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                found[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return found


@pytest.fixture
def repo_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    # receipts/ is the repository's immutable evidence store. data/ is derived scratch that
    # unrelated tests legitimately rewrite in parallel, so it is not an invariant of this run.
    watched = (_REPO / "receipts",)
    before = _tree_snapshot(*watched)
    monkeypatch.chdir(tmp_path)
    yield
    assert _tree_snapshot(*watched) == before


def _hermetic_config(tmp_path: Path, run_dir: Path) -> None:
    """Copy the configs under ``tmp_path`` and point ``data.root`` into the run dir."""
    destination = tmp_path / "configs"
    destination.mkdir()
    shutil.copy(_REPO / "configs" / "base.yaml", destination / "base.yaml")
    document = yaml.safe_load((_REPO / "configs" / "research.yaml").read_text(encoding="utf-8"))
    document.setdefault("data", {})["root"] = str(run_dir / "data")
    (destination / "research.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _roots(tmp_path: Path, run_dir: Path) -> dict[str, list[Path]]:
    (tmp_path / "input").mkdir(exist_ok=True)
    return {"config": [tmp_path], "input": [tmp_path / "input"], "output_dir": [run_dir]}


def _live_catalog(tmp_path: Path) -> CapabilityCatalog:
    loaded = load_catalog(Harness(), config_dir=tmp_path / "catalog-cache")
    assert loaded.reason is None, loaded.reason
    assert loaded.catalog is not None
    return loaded.catalog


def _real_run(
    tmp_path: Path,
    goal: str,
    model: Callable[[str], str] | None,
    *,
    selector: Callable[[str], dict[str, Any]] | None = None,
    plan_model: Callable[[str], str] | None = None,
) -> tuple[SuperpowerPlan, Any, JobLedger, Path]:
    run_dir = tmp_path / "run"
    _hermetic_config(tmp_path, run_dir)
    catalog = _live_catalog(tmp_path)
    roots = _roots(tmp_path, run_dir)
    plan = plan_goal(
        goal, catalog=catalog, roots=roots, family_selector=selector, model_fn=plan_model
    )
    ledger = JobLedger(tmp_path / "jobs")
    result = run_plan(
        plan,
        SuperpowerContext(
            model_fn=model,
            harness=Harness(),
            catalog=catalog,
            run_dir=run_dir,
            roots=roots,
            job_ledger=ledger,
            session_id=new_session_id(),
            run_id=new_run_id(),
        ),
    )
    return plan, result, ledger, run_dir


def _assert_clean_answer(summary: str) -> None:
    assert summary.count("Evidence (harness-verified") == 1
    assert validate_fx1_output(summary) == summary
    lowered = summary.lower()
    for token in _FORBIDDEN:
        assert token not in lowered
    assert " nav" not in lowered


def _step_argv(ledger: JobLedger, job_id: str | None) -> list[str]:
    job = ledger.get(job_id or "")
    assert job.state is JobState.SUCCEEDED
    assert len(job.steps) == 1
    return list(job.steps[0].argv_summary)


def test_real_catalog_vol_goal_runs_vol_bench_and_verifies_receipt(
    tmp_path: Path, repo_untouched: None
) -> None:
    def model(prompt: str) -> str:
        match = re.search(r"receipt=(\S+)", prompt)
        assert match is not None
        return f"SYNTHETIC QLIKE volatility result is a verified receipt {match.group(1)}"

    small = {
        "--models": "rv_roll",
        "--shards": "garch_vol",
        "--horizons": "1",
        "--min-history": 64,
        "--n-origins": 10,
        "--out-dir": str(tmp_path / "run"),
    }
    reply = json.dumps({"steps": [{"capability": "vol-bench", "args": small, "why": "QLIKE"}]})
    plan, result, ledger, run_dir = _real_run(
        tmp_path, "volatility forecast QLIKE scoring", model, plan_model=lambda _prompt: reply
    )
    assert plan.source == "model+validated"
    assert [step.capability_id for step in plan.typed_steps] == ["vol-bench"]
    assert plan.typed_steps[0].args["--out-dir"] == str(run_dir)
    assert result.status == "verified"
    verified = EvidenceLedger.from_dict(result.evidence).verified_receipts()
    assert len(verified) == 1 and verified[0].verdict is Verdict.VERIFIED
    assert verified[0].data_label == "SYNTHETIC"
    assert "verified receipt" in result.summary
    _assert_clean_answer(result.summary)
    argv = _step_argv(ledger, result.job_id)
    assert argv[0] == "vol-bench"
    assert argv[argv.index("--out-dir") + 1].startswith(str(run_dir))
    assert any(run_dir.rglob("vol_bench_*.json"))


def test_real_catalog_selective_goal_runs_research_only_as_partial_record(
    tmp_path: Path, repo_untouched: None
) -> None:
    runner = FamilyClient(harness_family_runner(Harness()))
    goal = "run only the conformal and evalues families"
    surface = runner.select(goal, max_families=4)
    chosen = [row["id"] for row in surface["selected"]]
    assert chosen == ["conformal", "evalues"]

    def model(prompt: str) -> str:
        match = re.search(r"json=(\S+)", prompt)
        assert match is not None
        return (
            "SYNTHETIC partial family record, verified=false and not a sealed "
            f"receipt: {match.group(1)}"
        )

    plan, result, ledger, run_dir = _real_run(
        tmp_path, goal, model, selector=lambda text: runner.select(text, max_families=4)
    )
    step = plan.typed_steps[0]
    assert [item.capability_id for item in plan.typed_steps] == ["research-only"]
    assert list(step.args["--only"]) == chosen
    assert plan.families == chosen
    argv = _step_argv(ledger, result.job_id)
    assert argv[:2] == ["research", "--config"]
    assert argv[argv.index("--only") + 1] == ",".join(chosen)
    assert "research-only" not in argv
    evidence = EvidenceLedger.from_dict(result.evidence)
    items = evidence.items()
    assert items
    assert all(item.verdict is not Verdict.VERIFIED for item in items)
    assert evidence.verified_receipts() == []
    assert any("verified=false" in item.detail for item in items)
    assert result.status == "unverified"
    assert "verified=false" in result.summary
    _assert_clean_answer(result.summary)
    assert any((run_dir / "data" / "family_runs").glob("*.json"))


def test_real_catalog_full_scorecard_goal_is_refused_under_the_family_selector(
    tmp_path: Path, repo_untouched: None
) -> None:
    _hermetic_config(tmp_path, tmp_path / "run")
    catalog = _live_catalog(tmp_path)
    roots = _roots(tmp_path, tmp_path / "run")
    runner = FamilyClient(harness_family_runner(Harness()))
    plan = plan_goal(
        "run the full scorecard",
        catalog=catalog,
        roots=roots,
        family_selector=lambda text: runner.select(text, max_families=4),
    )
    assert "research" not in [step.capability_id for step in plan.typed_steps]
    refused = [item for item in plan.exclusions if item["capability_id"] == "research"]
    assert refused and refused[0]["code"] == "not_executable"
    assert refused[0]["detail"]
    assert plan.research_constrained is True and plan.families == []


def test_real_catalog_only_comes_from_the_surface_not_goal_tokens(
    tmp_path: Path, repo_untouched: None
) -> None:
    _hermetic_config(tmp_path, tmp_path / "run")
    catalog = _live_catalog(tmp_path)
    runner = FamilyClient(harness_family_runner(Harness()))
    goal = "selective family run on variance forecast diagnostics"
    surface = runner.select(goal, max_families=4)
    chosen = [row["id"] for row in surface["selected"]]
    assert chosen and "variance" not in chosen
    selective = catalog.get("research-only")
    assert selective is not None
    only = next(spec for spec in selective.args if spec.name == "--only")
    assert set(chosen) <= set(only.choices or ())
    plan = plan_goal(
        goal,
        catalog=catalog,
        roots=_roots(tmp_path, tmp_path / "run"),
        family_selector=lambda text: runner.select(text, max_families=4),
    )
    bound = [step for step in plan.typed_steps if step.capability_id == "research-only"]
    assert len(bound) == 1
    assert list(bound[0].args["--only"]) == chosen
    assert plan.families == chosen


def _failing_harness() -> Any:
    class Failing:
        def run(self, name: str, extra: list[str] | None = None, **_kw: Any) -> SimpleNamespace:
            assert (name, extra) == ("capabilities", ["list", "--json"])
            return SimpleNamespace(exit_code=1, stdout="", stderr="boom token=sk-secret123456789")

    return Failing()


def test_failed_catalog_fetch_fails_closed_with_visible_reason(tmp_path: Path) -> None:
    loaded = load_catalog(_failing_harness(), config_dir=tmp_path / "cache")
    assert loaded.catalog is None
    assert loaded.reason
    assert "sk-secret123456789" not in loaded.reason
    plan = plan_goal("run vol bench", catalog_unavailable=loaded.reason)
    assert plan.typed_steps == ()
    assert plan.catalog_unavailable
    assert loaded.reason[:20] in plan.describe()
    harness = _RecordingHarness()
    result = run_plan(plan, SuperpowerContext(harness=harness, run_dir=tmp_path / "run"))
    assert harness.calls == []
    assert result.status != "verified"


class _RecordingHarness:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, name: str, *_a: Any, **_kw: Any) -> SimpleNamespace:
        self.calls.append(name)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def test_qualified_capability_outside_harness_registry_is_excluded(tmp_path: Path) -> None:
    document = json.loads(
        Harness().run("capabilities", ["list", "--json"]).stdout  # live producer output
    )
    ghost = json.loads(
        json.dumps(next(cap for cap in document["capabilities"] if cap["id"] == "vol-bench"))
    )
    ghost.update({"id": "ghost-bench", "aliases": ["ghost bench", "ghost volatility"]})
    ghost["command"]["argv"] = ["ghost-bench"]
    document["capabilities"].append(ghost)
    catalog = CapabilityCatalog.from_json(document)
    assert catalog.get("ghost-bench") is not None
    assert catalog.get("ghost-bench").executable is True  # type: ignore[union-attr]
    run_dir = tmp_path / "run"
    _hermetic_config(tmp_path, run_dir)
    plan = plan_goal("run ghost bench", catalog=catalog, roots=_roots(tmp_path, run_dir))
    assert "ghost-bench" not in [step.capability_id for step in plan.typed_steps]
    assert any("not registered in HARNESS_REGISTRY" in item["detail"] for item in plan.exclusions)
    assert "not registered in HARNESS_REGISTRY" in plan.describe()
