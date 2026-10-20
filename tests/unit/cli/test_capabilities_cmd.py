"""Lab capability contract: fail closed, deterministic, and schema-stable."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.capability_cmds import (
    FAIL_CLOSED_REASON,
    FamilyCensus,
    LabCommand,
    audit_capabilities,
    audit_family_contract,
    build_capability_document,
    build_family_document,
    commands_from_app,
    dumps_document,
    list_lab_commands,
)
from quant_fund.cli.capability_overlay import CURATED_OVERLAY, overlay_sha256, qualified_ids
from quant_fund.cli.main import app

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_EVIDENCE = "tests/unit/cli/test_capabilities_cmd.py"
_RUNNER = CliRunner()


def _env(tmp: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC) + (os.pathsep + previous if previous else "")
    if tmp is not None:
        env["FX1_CONFIG_DIR"] = str(tmp)
    return env


def _arg(
    name: str,
    *,
    kind: str = "option",
    typ: str = "str",
    required: bool = False,
    default: str | None = None,
    path_role: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "kind": kind,
        "type": "path" if path_role else typ,
        "required": required,
        "default": default,
        "repeatable": False,
        "help": "",
    }
    if path_role:
        payload["path_role"] = path_role
    return payload


def _command(
    *argv: str, args: tuple[dict[str, object], ...] = (), help_text: str = "Help."
) -> LabCommand:
    return LabCommand(argv=argv, help_text=help_text, hidden=False, args=args)


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "demo",
        "version": "1",
        "purpose": "Demo capability.",
        "tasks": ["demo_task"],
        "aliases": ["demo"],
        "command": {
            "argv": ["demo"],
            "role": "verification",
            "read_only": True,
            "writes": [],
            "consequential": False,
            "timeout_s": 30,
        },
        "allowed_args": ["--config"],
        "arg_hints": {},
        "inputs": {
            "data_labels": [],
            "requires_dataset": False,
            "availability_timing": "n/a",
        },
        "outputs": {
            "scores": [],
            "receipt_schema": "",
            "receipt_flag": [],
            "receipt_stdout_marker": "",
        },
        "dependencies": [],
        "post_checks": [{"kind": "exit_code"}],
        "limits": {"max_runtime_s": 30, "max_output_chars": 1000},
        "qualification": {
            "status": "qualified",
            "evidence": [_EVIDENCE],
            "ruleset": "overlay-v1",
            "audited_on": "2026-10-12",
        },
        "evidence_class": "VERIFICATION",
    }
    entry.update(overrides)
    return entry


def _demo() -> LabCommand:
    return _command(
        "demo", args=(_arg("--config", path_role="config", default="configs/research.yaml"),)
    )


def _by_id() -> dict[str, dict[str, object]]:
    doc = build_capability_document()
    return {str(cap["id"]): cap for cap in doc["capabilities"]}


def _run(args: list[str], tmp: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=_ROOT,
        env=_env(tmp),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_capability_module_body_does_not_import_registry() -> None:
    tree = ast.parse((_SRC / "quant_fund/cli/capability_cmds.py").read_text())
    modules = [
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert "quant_fund.research.catalog.registry" not in modules
    assert "quant_fund.research.family_surface" not in modules
    assert "quant_fund.cli.main" not in modules
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_family_census"
    )
    nested = [node.module for node in ast.walk(loader) if isinstance(node, ast.ImportFrom)]
    assert "quant_fund.research.catalog.registry" in nested


def test_import_stays_lazy_until_families_are_loaded() -> None:
    """Package ``quant_fund.cli`` imports main; stub it so this asserts our module."""
    script = """
import sys
import types
pkg = types.ModuleType("quant_fund.cli")
pkg.__path__ = ["src/quant_fund/cli"]
pkg.__package__ = "quant_fund.cli"
sys.modules["quant_fund.cli"] = pkg
import quant_fund.cli.capability_cmds as cmds
assert "quant_fund.research.catalog.registry" not in sys.modules
assert "quant_fund.research.family_surface" not in sys.modules
assert "quant_fund.cli.main" not in sys.modules
cmds.load_family_census()
assert "quant_fund.research.catalog.registry" in sys.modules
assert "quant_fund.research.family_surface" not in sys.modules
"""
    completed = _run(["-c", script], Path("."))
    assert completed.returncode == 0, completed.stderr


def test_overlay_sha_is_stable_hex() -> None:
    first = overlay_sha256()
    second = overlay_sha256(CURATED_OVERLAY)
    assert first == second
    assert len(first) == 64
    assert first == first.lower()


def test_list_document_schema_and_determinism() -> None:
    first = build_capability_document()
    second = build_capability_document()
    assert dumps_document(first) == dumps_document(second)
    assert first["schema"] == "dipcatcher.capabilities.v1"
    assert first["generated_by"]["overlay_sha256"] == overlay_sha256()
    assert first["generated_by"]["n_lab_commands"] == len(list_lab_commands())
    ids = [cap["id"] for cap in first["capabilities"]]
    command_ids = {cmd.id for cmd in list_lab_commands()}
    assert ids == sorted(ids)
    assert command_ids <= set(ids)
    assert [ident for ident in ids if ident not in command_ids] == ["research-only"]
    assert len(ids) == first["generated_by"]["n_lab_commands"] + 1


def test_executable_ids_match_the_curated_overlay() -> None:
    doc = build_capability_document(executable_only=True)
    assert [cap["id"] for cap in doc["capabilities"]] == sorted(qualified_ids())
    assert doc["generated_by"]["n_lab_commands"] > len(doc["capabilities"])
    assert all(cap["qualification"]["status"] == "qualified" for cap in doc["capabilities"])
    assert all(cap["executable_reason"] == "" for cap in doc["capabilities"])


def test_commands_outside_the_overlay_fail_closed() -> None:
    ingest = _by_id()["ingest"]
    assert ingest["qualification"]["status"] == "unqualified"
    assert ingest["executable"] is False
    assert ingest["executable_reason"] == FAIL_CLOSED_REASON
    assert ingest["version"] == "0"


def test_consequential_commands_stay_unqualified() -> None:
    caps = _by_id()
    for ident in ("paper", "optimize", "train volatility", "hmm train"):
        cap = caps[ident]
        assert cap["command"]["consequential"] is True
        assert cap["executable"] is False
        assert cap["qualification"]["status"] == "unqualified"


def test_hidden_lab_alias_is_listed_and_not_executable() -> None:
    lab = _by_id()["lab"]
    assert lab["executable"] is False
    assert lab["qualification"]["status"] == "unqualified"


def test_vol_bench_publishes_csv_horizons_and_receipt_v2() -> None:
    cap = _by_id()["vol-bench"]
    by_name = {arg["name"]: arg for arg in cap["args"]}
    horizons = by_name["--horizons"]
    assert horizons["type"] == "csv"
    assert horizons["item_type"] == "int"
    assert horizons["default"] == "1,5"
    version = by_name["--receipt-version"]
    assert version["default"] == 2
    assert version["choices"] == ["2"]
    assert version["min"] == 2
    assert version["max"] == 2
    assert cap["outputs"]["receipt_flag"] == ["--receipt-version", "2"]
    assert cap["outputs"]["receipt_stdout_marker"] == "receipt="
    assert cap["outputs"]["scores"] == ["QLIKE", "MSE"]
    assert cap["evidence_class"] == "SYNTHETIC"
    assert cap["inputs"]["data_labels"] == ["SYNTHETIC"]


def test_serial_watch_forces_synthetic_label() -> None:
    cap = _by_id()["serial-watch"]
    label = next(arg for arg in cap["args"] if arg["name"] == "--data-label")
    assert label["type"] == "enum"
    assert label["choices"] == ["SYNTHETIC"]
    assert label["default"] == "SYNTHETIC"
    pits = next(arg for arg in cap["args"] if arg["name"] == "pits")
    assert pits["kind"] == "argument"
    assert pits["path_role"] == "input"
    assert pits["required"] is True


def test_verify_all_does_not_expose_weakening_flags() -> None:
    names = {arg["name"] for arg in _by_id()["verify-all"]["args"]}
    assert names == {"--receipts-dir", "--out-dir"}
    assert "--check-index" not in names
    assert "--write" not in names
    out = next(arg for arg in _by_id()["verify-all"]["args"] if arg["name"] == "--out-dir")
    assert out["path_role"] == "output_dir"


def test_research_full_scorecard_is_separate_from_selective_runs() -> None:
    cap = _by_id()["research"]
    assert "REQUIRED" in str(cap["purpose"])
    assert "cannot be narrowed" not in str(cap["purpose"])
    assert cap["command"]["argv"] == ["research"]
    assert [arg["name"] for arg in cap["args"]] == ["--config"]
    assert cap["post_checks"] == [{"kind": "verify_research", "from": "json="}]
    assert cap["outputs"]["receipt_stdout_marker"] == "json="
    assert cap["limits"]["max_runtime_s"] == 3600
    assert cap["evidence_class"] == "SYNTHETIC"
    assert cap["executable"] is True
    assert cap["version"] == "2"
    assert cap["qualification"]["audited_on"] == "2026-10-12"


def test_path_roles_and_dropped_url_defaults() -> None:
    caps = _by_id()
    config = next(arg for arg in caps["doctor"]["args"] if arg["name"] == "--config")
    assert config["path_role"] == "config"
    assert config["default"] == "configs/research.yaml"
    url = next(arg for arg in caps["anchor-timestamp"]["args"] if arg["name"] == "--tsr-url")
    assert url["default"] is None
    text = dumps_document(build_capability_document())
    assert "/home" not in text
    assert "https://" not in text
    assert "freetsa" not in text
    assert "rekor.sigstore" not in text


def test_csv_is_not_inferred_from_a_comma_default() -> None:
    obs = next(arg for arg in _by_id()["hmm train"]["args"] if arg["name"] == "--obs")
    assert obs["type"] == "str"
    assert "item_type" not in obs
    assert obs["default"] == "3,1,3,2,1,2,3"


def test_repeatable_options_are_marked() -> None:
    bars = next(arg for arg in _by_id()["membership-coverage"]["args"] if arg["name"] == "--bars")
    assert bars["repeatable"] is True
    assert bars["type"] == "path"


def test_every_executable_arg_exists_on_the_real_command() -> None:
    commands = {cmd.id: cmd for cmd in list_lab_commands()}
    for cap in build_capability_document()["capabilities"]:
        if not cap["executable"]:
            continue
        key = " ".join(str(part) for part in cap["command"]["argv"])
        real = {arg["name"] for arg in commands[key].args}
        for arg in cap["args"]:
            assert arg["name"] in real


def test_human_list_and_describe_fit_100_columns() -> None:
    listed = _RUNNER.invoke(app, ["capabilities", "list"])
    assert listed.exit_code == 0, listed.output
    assert listed.stdout.splitlines()
    assert all(len(line) <= 100 for line in listed.stdout.splitlines())
    described = _RUNNER.invoke(app, ["capabilities", "describe", "vol-bench"])
    assert described.exit_code == 0, described.output
    assert all(len(line) <= 100 for line in described.stdout.splitlines())
    assert "vol-bench" in described.stdout


def test_cli_list_json_is_one_document() -> None:
    result = _RUNNER.invoke(app, ["capabilities", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema"] == "dipcatcher.capabilities.v1"
    assert isinstance(payload["capabilities"], list)


def test_cli_executable_only_matches_overlay() -> None:
    result = _RUNNER.invoke(app, ["capabilities", "list", "--executable-only", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [cap["id"] for cap in payload["capabilities"]] == sorted(qualified_ids())


def test_cli_json_invocations_are_byte_identical() -> None:
    first = _RUNNER.invoke(app, ["capabilities", "list", "--json"])
    second = _RUNNER.invoke(app, ["capabilities", "list", "--json"])
    assert first.exit_code == 0
    assert first.stdout == second.stdout


def test_describe_unknown_id_fails_closed() -> None:
    result = _RUNNER.invoke(app, ["capabilities", "describe", "not-a-capability", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown capability"
    assert payload["capability"] is None


def test_families_paging_on_an_injected_census() -> None:
    census = FamilyCensus(
        required=("ranking", "alpha"),
        live=("candle_order_book",),
        retired=(("old_family", "template stub", 2),),
    )
    doc = build_family_document(status="all", limit=2, offset=0, census=census)
    assert doc["schema"] == "dipcatcher.capabilities.v1"
    assert doc["total"] == 4
    assert doc["next_offset"] == 2
    assert doc["counts"] == {"required": 2, "live": 1, "retired": 1}
    assert [row["id"] for row in doc["families"]] == ["ranking", "alpha"]
    assert doc["families"][0]["executable"] is True
    assert doc["families"][0]["runner"] == "research --only ranking"
    assert doc["families"][0]["executable_reason"] == ""
    assert doc["families"][0]["resource_limit_s"] > 0
    assert doc["families"][0]["purpose"]
    assert doc["families"][1]["runner"] == "research --only alpha"
    live = build_family_document(status="live", limit=10, offset=0, census=census)
    assert live["families"][0]["id"] == "candle_order_book"
    assert live["families"][0]["executable"] is False
    assert live["families"][0]["runner"] is None
    assert "live-optional" in str(live["families"][0]["executable_reason"])
    retired = build_family_document(status="retired", limit=10, offset=0, census=census)
    assert retired["families"][0]["retired_reason"] == "template stub"
    assert retired["families"][0]["last_receipt_schema_version"] == 2
    assert retired["families"][0]["executable"] is False
    assert "retired" in str(retired["families"][0]["executable_reason"])
    assert audit_family_contract(census) == []


def test_family_page_rejects_bad_bounds() -> None:
    census = FamilyCensus(required=(), live=(), retired=())
    with pytest.raises(ValueError, match="status"):
        build_family_document(status="proposed", census=census)
    with pytest.raises(ValueError, match="limit"):
        build_family_document(limit=0, census=census)
    with pytest.raises(ValueError, match="offset"):
        build_family_document(offset=-1, census=census)
    empty = build_family_document(limit=5, offset=3, census=census)
    assert empty["families"] == []
    assert empty["next_offset"] is None


def test_cli_families_reject_unknown_status() -> None:
    result = _RUNNER.invoke(app, ["capabilities", "families", "--status", "proposed"])
    assert result.exit_code != 0


def test_real_families_keep_catalog_order_and_runner_truth() -> None:
    from quant_fund.research.catalog.registry import (
        BENCHMARK_FAMILY_ORDER,
        LIVE_OPTIONAL_BENCHMARK_FAMILIES,
        REQUIRED_BENCHMARK_FAMILIES,
        RETIRED_BENCHMARK_FAMILIES,
    )
    from quant_fund.research.family_surface import RUNNERS, family_contract

    required = build_family_document(status="required", limit=50, offset=0)
    assert [row["id"] for row in required["families"]] == list(BENCHMARK_FAMILY_ORDER)
    assert required["counts"]["required"] == len(REQUIRED_BENCHMARK_FAMILIES)
    assert required["counts"]["live"] == len(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    assert required["counts"]["retired"] == len(RETIRED_BENCHMARK_FAMILIES)
    for row in required["families"]:
        if row["id"] in RUNNERS:
            assert row["executable"] is True
            assert row["runner"] == f"research --only {row['id']}"
            assert row["executable_reason"] == ""
        else:
            assert row["executable"] is False
            assert row["runner"] is None
    live = build_family_document(status="live", limit=1, offset=0)
    live_row = live["families"][0]
    assert live_row["executable"] is False
    assert live_row["executable_reason"] == family_contract(str(live_row["id"]))["refusal"]
    retired = build_family_document(status="retired", limit=1, offset=0)
    assert retired["total"] == len(RETIRED_BENCHMARK_FAMILIES)
    assert retired["next_offset"] == 1
    retired_row = retired["families"][0]
    assert retired_row["executable"] is False
    assert retired_row["retired_reason"]
    assert isinstance(retired_row["last_receipt_schema_version"], int)
    assert retired_row["executable_reason"] == family_contract(str(retired_row["id"]))["refusal"]


def test_family_human_lines_fit_100_columns() -> None:
    result = _RUNNER.invoke(
        app, ["capabilities", "families", "--status", "required", "--limit", "5"]
    )
    assert result.exit_code == 0, result.output
    assert all(len(line) <= 100 for line in result.stdout.splitlines())
    assert "ranking" in result.stdout


def test_check_passes_on_the_real_tree() -> None:
    assert audit_capabilities(include_families=True) == []
    result = _RUNNER.invoke(app, ["capabilities", "check", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["problems"] == []
    assert all(
        len(line) <= 100
        for line in _RUNNER.invoke(app, ["capabilities", "check"]).stdout.splitlines()
    )


def test_check_fails_on_a_duplicate_overlay_id() -> None:
    entry = _entry()
    problems = audit_capabilities(overlay=[entry, entry], commands=[_demo()], discover_repo=True)
    assert any(problem.startswith("duplicate capability id: demo") for problem in problems)


def test_check_fails_on_an_unknown_command() -> None:
    entry = _entry(
        id="not-real",
        command={
            "argv": ["not-real"],
            "role": "verification",
            "read_only": True,
            "writes": [],
            "consequential": False,
            "timeout_s": 30,
        },
    )
    problems = audit_capabilities(overlay=[entry], commands=[_demo()], discover_repo=False)
    assert any("command.argv[0] is not a real lab command" in problem for problem in problems)
    assert any("command argv is not a real lab command" in problem for problem in problems)


def test_check_fails_when_evidence_is_missing(tmp_path: Path) -> None:
    problems = audit_capabilities(
        overlay=[_entry()],
        commands=[_demo()],
        repo_root=tmp_path,
        discover_repo=False,
    )
    assert any("qualification evidence missing:" in problem for problem in problems)
    doc = build_capability_document(
        overlay=[_entry()],
        commands=[_demo()],
        repo_root=tmp_path,
        discover_repo=False,
    )
    assert doc["capabilities"][0]["executable"] is False


def test_check_fails_on_an_unknown_arg_hint() -> None:
    entry = _entry(arg_hints={"--nope": {"max": 1}})
    problems = audit_capabilities(overlay=[entry], commands=[_demo()], discover_repo=False)
    assert any("unknown arg hint: --nope" in problem for problem in problems)


def test_check_fails_when_executable_is_forced_for_a_non_qualified_status() -> None:
    qual = {
        "status": "experimental",
        "evidence": [],
        "ruleset": "overlay-v1",
        "audited_on": "2026-10-12",
    }
    entry = _entry(qualification=qual, force_executable=True)
    problems = audit_capabilities(overlay=[entry], commands=[_demo()], discover_repo=False)
    assert any("executable with non-qualified status" in problem for problem in problems)


def test_retired_overlay_is_not_executable_unless_forced() -> None:
    qual = {
        "status": "retired",
        "evidence": [],
        "ruleset": "overlay-v1",
        "audited_on": "2026-10-12",
    }
    clean = audit_capabilities(
        overlay=[_entry(qualification=qual)],
        commands=[_demo()],
        discover_repo=False,
    )
    assert clean == []
    forced = audit_capabilities(
        overlay=[_entry(qualification=qual, force_executable=True)],
        commands=[_demo()],
        discover_repo=False,
    )
    assert any("retired capability is executable" in problem for problem in forced)
    doc = build_capability_document(
        overlay=[_entry(qualification=qual)],
        commands=[_demo()],
        discover_repo=False,
    )
    assert doc["capabilities"][0]["executable"] is False


def test_missing_repo_syntax_checks_evidence_without_reading_files() -> None:
    qual = {
        "status": "qualified",
        "evidence": ["tests/missing_but_well_formed.py"],
        "ruleset": "overlay-v1",
        "audited_on": "2026-10-12",
    }
    doc = build_capability_document(
        overlay=[_entry(qualification=qual)],
        commands=[_demo()],
        discover_repo=False,
    )
    assert doc["capabilities"][0]["executable"] is True
    assert (
        audit_capabilities(
            overlay=[_entry(qualification=qual)],
            commands=[_demo()],
            discover_repo=False,
        )
        == []
    )


def test_parent_traversal_evidence_is_rejected() -> None:
    qual = {
        "status": "qualified",
        "evidence": ["../secrets.py"],
        "ruleset": "overlay-v1",
        "audited_on": "2026-10-12",
    }
    problems = audit_capabilities(
        overlay=[_entry(qualification=qual)],
        commands=[_demo()],
        discover_repo=False,
    )
    assert any("not a relative test file" in problem for problem in problems)


def test_weakening_flag_and_loosened_max_fail_the_audit() -> None:
    command = _command(
        "demo",
        args=(
            _arg("--dev", typ="flag", default="false"),
            _arg("--n", typ="int", default="5"),
        ),
    )
    weakening = _entry(allowed_args=["--dev"])
    assert any(
        "weakening flag" in problem
        for problem in audit_capabilities(
            overlay=[weakening], commands=[command], discover_repo=False
        )
    )
    published = build_capability_document(
        overlay=[weakening], commands=[command], discover_repo=False
    )
    assert all(arg["name"] != "--dev" for arg in published["capabilities"][0]["args"])
    loosened = _entry(allowed_args=["--n"], arg_hints={"--n": {"max": 9}})
    command.args[1]["max"] = 5
    problems = audit_capabilities(overlay=[loosened], commands=[command], discover_repo=False)
    assert any("overlay loosens max" in problem for problem in problems)


def test_introspection_drops_host_paths_and_reads_choices() -> None:
    import click

    @click.command("demo")
    @click.option("--horizons", default="1,5", help="Comma  separated   horizons.\nignored")
    @click.option("--choice", type=click.Choice(["a", "b"]), default="a")
    @click.option("--counted", type=click.IntRange(1, 5), default=2)
    @click.option("--out-dir", type=click.Path(), default="/home/me/out")
    @click.option("--url", default="https://example.com/x")
    @click.option("--win", default="C:/Users/me")
    @click.option("--flag", is_flag=True, default=False)
    def demo() -> None:
        """Demo help line."""

    commands = {cmd.id: cmd for cmd in commands_from_app(demo)}
    args = {arg["name"]: arg for arg in commands["demo"].args}
    assert args["--horizons"]["type"] == "str"
    assert args["--horizons"]["help"] == "Comma separated horizons."
    assert args["--choice"]["type"] == "enum"
    assert args["--choice"]["choices"] == ["a", "b"]
    assert args["--counted"]["min"] == 1
    assert args["--counted"]["max"] == 5
    assert args["--out-dir"]["default"] is None
    assert args["--out-dir"]["path_role"] == "output_dir"
    assert args["--url"]["default"] is None
    assert args["--win"]["default"] is None
    assert args["--flag"]["type"] == "flag"
    assert args["--flag"]["default"] == "false"
    blob = json.dumps(commands["demo"].args)
    assert "/home" not in blob
    assert "example.com" not in blob
    assert "C:/" not in blob


def test_subprocess_list_json_matches_the_schema(tmp_path: Path) -> None:
    completed = _run(["-m", "dipcatcher_cli", "capabilities", "list", "--json"], tmp_path)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "dipcatcher.capabilities.v1"
    assert payload["generated_by"]["overlay_sha256"] == overlay_sha256()
    assert any(cap["id"] == "vol-bench" and cap["executable"] for cap in payload["capabilities"])


def test_group_help_lists_the_subcommands() -> None:
    result = _RUNNER.invoke(app, ["capabilities", "--help"])
    assert result.exit_code == 0
    for name in ("list", "describe", "families", "check"):
        assert name in result.stdout


_RECEIPT_BENCHES = ("vol-bench", "calibration-eval", "rankic", "serial-watch", "lane-power")


def _targets_receipts(default: object) -> bool:
    if not isinstance(default, str) or not default.strip():
        return False
    parts = [
        part.lower() for part in default.replace("\\", "/").split("/") if part not in {"", "."}
    ]
    return "receipts" in parts


def test_research_only_publishes_a_partial_unverified_record() -> None:
    from quant_fund.research.family_surface import RUNNERS, list_executable

    cap = _by_id()["research-only"]
    assert cap["executable"] is True
    assert cap["version"] == "1"
    assert cap["command"]["argv"] == ["research"]
    assert cap["evidence_class"] == "SYNTHETIC"
    assert cap["outputs"]["receipt_schema"] == "dipcatcher.family_run.v1"
    assert cap["outputs"]["receipt_stdout_marker"] == "json="
    assert cap["post_checks"] == []
    assert "verified" in str(cap["purpose"])
    only = next(arg for arg in cap["args"] if arg["name"] == "--only")
    assert only["type"] == "csv"
    assert only["item_type"] == "str"
    assert only["required"] is True
    assert only["choices"] == [row["id"] for row in list_executable()]
    assert set(only["choices"]) <= set(RUNNERS)
    config = next(arg for arg in cap["args"] if arg["name"] == "--config")
    assert config["path_role"] == "config"


def test_only_choices_match_executable_families() -> None:
    from quant_fund.research.family_surface import RUNNERS

    census_total = build_family_document(status="all", limit=1, offset=0)["total"]
    families = build_family_document(status="all", limit=census_total, offset=0)
    executable = [row["id"] for row in families["families"] if row["executable"]]
    only = next(arg for arg in _by_id()["research-only"]["args"] if arg["name"] == "--only")
    assert only["choices"] == executable
    assert executable
    assert all(ident in RUNNERS for ident in executable)
    assert all(row["status"] == "required" for row in families["families"] if row["executable"])
    assert all(
        row["executable"] is False for row in families["families"] if row["status"] != "required"
    )


def test_verify_all_is_unqualified_because_it_writes_receipts() -> None:
    cap = _by_id()["verify-all"]
    assert cap["qualification"]["status"] == "unqualified"
    assert cap["executable"] is False
    assert cap["version"] == "2"
    assert "receipts/" in str(cap["executable_reason"])
    assert "unqualified capability is not executable" not in str(cap["executable_reason"])


def test_receipt_benches_require_an_explicit_out_dir() -> None:
    caps = _by_id()
    for ident in _RECEIPT_BENCHES:
        arg = next(item for item in caps[ident]["args"] if item["name"] == "--out-dir")
        assert arg["path_role"] == "output_dir"
        assert arg["default"] is None
        assert arg["required"] is True
        assert caps[ident]["executable"] is True
        assert caps[ident]["version"] == "2"


def test_executable_capabilities_do_not_default_into_receipts() -> None:
    for cap in build_capability_document()["capabilities"]:
        if not cap["executable"]:
            continue
        for arg in cap["args"]:
            assert not _targets_receipts(arg.get("default")), (cap["id"], arg)


def test_executable_argv0_is_registered_in_the_harness() -> None:
    from fx1.harness import HARNESS_REGISTRY

    names = {command.name for command in HARNESS_REGISTRY}
    missing = [
        (cap["id"], cap["command"]["argv"][0])
        for cap in build_capability_document(executable_only=True)["capabilities"]
        if cap["command"]["argv"][0] not in names
    ]
    assert missing == []
    for name in ("research", "families", "doctor", "vol-bench", "verify-research"):
        assert name in names


def test_doctor_in_an_empty_tree_reports_absence_and_creates_nothing(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "doctor",
            "--config",
            str(_ROOT / "configs" / "research.yaml"),
        ],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 1, completed.stderr
    assert "dir_raw: missing" in completed.stdout
    files = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert files == []
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "receipts").exists()
    cap = _by_id()["doctor"]
    assert cap["executable"] is True
    assert cap["qualification"]["status"] == "qualified"
    assert cap["command"]["writes"] == ["resolved_config"]


def test_surface_import_waits_until_a_capability_document_is_built(tmp_path: Path) -> None:
    script = """
import sys
import quant_fund.cli.capability_cmds as cmds
assert "quant_fund.research.family_surface" not in sys.modules
cmds.build_capability_document()
assert "quant_fund.research.family_surface" in sys.modules
"""
    completed = _run(["-c", script], tmp_path)
    assert completed.returncode == 0, completed.stderr
