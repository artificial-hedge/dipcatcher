"""Live capability contract: real Click commands, real receipts, real client."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fx1.interactive.contracts import (
    SCHEMA_ID,
    Capability,
    CapabilityCatalog,
    ContractError,
    build_argv,
    validate_args,
)
from quant_fund.cli.capability_cmds import commands_from_app
from quant_fund.cli.capability_overlay import CURATED_OVERLAY, EVIDENCE_CLASSES
from quant_fund.cli.main import app

_ROOT = Path(__file__).resolve().parents[3]
_RECEIPT = _ROOT / "receipts" / "alert_budget.json"


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(_ROOT / "src"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "FX1_CONFIG_DIR": str(tmp_path / "fx1"),
        "FX1_SESSION_DIR": str(tmp_path / "sessions"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        **extra,
    }


def _lab(tmp_path: Path, args: list[str], **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", *args],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=_env(tmp_path, **extra),
        timeout=60,
        check=False,
    )


def _commands() -> dict[str, object]:
    return {command.id: command for command in commands_from_app(app)}


def _independent_problems() -> list[str]:
    """Re-derive the overlay contract from Click, without calling the auditor."""
    commands = _commands()
    problems: list[str] = []
    for entry in CURATED_OVERLAY:
        ident = str(entry["id"])
        argv = " ".join(entry["command"]["argv"])
        command = commands.get(argv)
        if command is None:
            problems.append(f"{ident}: command is not a real lab command")
            continue
        names = {str(arg["name"]) for arg in command.args}  # type: ignore[attr-defined]
        for flag in entry["allowed_args"]:
            if flag not in names:
                problems.append(f"{ident}: allowed_args {flag} is not a Click parameter")
        for name, hint in entry["arg_hints"].items():
            if "default" not in hint:
                continue
            default = hint["default"]
            try:
                number = float(default)
            except (TypeError, ValueError):
                number = None
            if number is None:
                continue
            if "min" in hint and number < float(hint["min"]):
                problems.append(f"{ident}: overlay default for {name} is below its min")
            if "max" in hint and number > float(hint["max"]):
                problems.append(f"{ident}: overlay default for {name} is above its max")
        if entry["evidence_class"] not in EVIDENCE_CLASSES:
            problems.append(f"{ident}: evidence class is not permitted")
        timeout_s = entry["command"]["timeout_s"]
        limit_s = entry["limits"]["max_runtime_s"]
        if timeout_s != limit_s or timeout_s < 1:
            problems.append(f"{ident}: timeout_s and max_runtime_s disagree")
        for rel in entry["qualification"]["evidence"]:
            if not (_ROOT / rel).is_file():
                problems.append(f"{ident}: evidence file is missing: {rel}")
    return problems


# Direct expert CLI use is outside the qualified window. These Click defaults
# differ from the overlay. A planned invocation does not rely on them: the
# catalog publishes the overlay default and ``build_argv`` passes it explicitly.
_CLICK_OUTSIDE_QUALIFIED_WINDOW = (
    "calibration-eval --receipt-version",
    "lane-power --n-seeds",
    "lane-power --n-steps",
    "lane-power --receipt-version",
    "rankic --receipt-version",
    "serial-watch --data-label",
    "serial-watch --receipt-version",
    "vol-bench --receipt-version",
)


def _cli_default_violations() -> list[str]:
    commands = _commands()
    violations: list[str] = []
    for entry in CURATED_OVERLAY:
        argv = " ".join(entry["command"]["argv"])
        command = commands[argv]
        by_name = {str(arg["name"]): arg for arg in command.args}  # type: ignore[attr-defined]
        for name, hint in entry["arg_hints"].items():
            arg = by_name[name]
            lo = hint.get("min", arg.get("min"))
            hi = hint.get("max", arg.get("max"))
            raw = arg.get("default")
            if raw is None or arg.get("type") not in {"int", "float"}:
                if "choices" in hint and raw is not None and str(raw) not in set(hint["choices"]):
                    violations.append(f"{entry['id']} {name}")
                continue
            number = float(raw)
            below = lo is not None and number < float(lo)
            above = hi is not None and number > float(hi)
            if below or above:
                violations.append(f"{entry['id']} {name}")
    return violations


def _sample_value(spec: object) -> object:
    choices = getattr(spec, "choices", None)
    kind = getattr(spec, "type", "")
    if choices:
        raw = choices[0]
        return int(raw) if kind == "int" else raw
    if kind == "int":
        minimum = getattr(spec, "minimum", None)
        return int(minimum if minimum is not None else 0)
    if kind == "float":
        minimum = getattr(spec, "minimum", None)
        return float(minimum if minimum is not None else 0.0)
    if kind == "flag":
        return False
    if kind == "path":
        role = getattr(spec, "path_role", None)
        return "out" if role == "output_dir" else "receipts/sample.json"
    if kind == "csv":
        item = getattr(spec, "item_type", None)
        if item == "int":
            return "1"
        if item == "float":
            return "0.0"
        return "sample"
    return "sample"


def _plan_inputs(cap: Capability, root: Path) -> tuple[dict[str, object], dict[str, list[Path]]]:
    roots = {role: [root / role] for role in ("input", "config", "output_dir", "any")}
    for directory in roots.values():
        directory[0].mkdir(parents=True, exist_ok=True)
    provided: dict[str, object] = {}
    for spec in cap.args:
        if not spec.required or spec.has_default:
            continue
        if spec.type == "path":
            role = spec.path_role or "any"
            target = roots[role][0] / "sample.json"
            target.write_text("{}", encoding="utf-8")
            provided[spec.name] = str(target)
            continue
        provided[spec.name] = _sample_value(spec)
    return provided, roots


def _outside_overlay(value: object, hint: dict[str, object]) -> str | None:
    if isinstance(value, list):
        for item in value:
            problem = _outside_overlay(item, hint)
            if problem:
                return problem
        return None
    choices = hint.get("choices")
    if isinstance(choices, list) and str(value) not in {str(item) for item in choices}:
        return f"choice {value!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if "min" in hint and float(value) < float(hint["min"]):  # type: ignore[arg-type]
        return f"{value} < min {hint['min']}"
    if "max" in hint and float(value) > float(hint["max"]):  # type: ignore[arg-type]
        return f"{value} > max {hint['max']}"
    return None


def _required_pins(entry: dict[str, object]) -> list[tuple[str, str]]:
    hints = entry["arg_hints"]
    pins: list[tuple[str, str]] = []
    if not isinstance(hints, dict):
        return pins
    receipt = hints.get("--receipt-version")
    if isinstance(receipt, dict) and (
        receipt.get("choices") == ["2"] or receipt.get("default") == "2"
    ):
        pins.append(("--receipt-version", "2"))
    label = hints.get("--data-label")
    if isinstance(label, dict) and (
        label.get("choices") == ["SYNTHETIC"] or label.get("default") == "SYNTHETIC"
    ):
        pins.append(("--data-label", "SYNTHETIC"))
    return pins


def _argv_has_pin(argv: list[str], flag: str, value: str) -> bool:
    return any(argv[index] == flag and argv[index + 1] == value for index in range(len(argv) - 1))


def _evidence_gaps() -> list[str]:
    gaps: list[str] = []
    for entry in CURATED_OVERLAY:
        token = entry["command"]["argv"][0]
        named = False
        for rel in entry["qualification"]["evidence"]:
            text = (_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            if token in text:
                named = True
        if not named:
            gaps.append(f"{entry['id']}: no evidence file mentions {token}")
    return gaps


def _receipt_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("receipt="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(stdout)


def test_capabilities_check_agrees_with_an_independent_contract(tmp_path: Path) -> None:
    mine = _independent_problems()
    proc = _lab(tmp_path, ["capabilities", "check", "--json"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    reported = json.loads(proc.stdout)
    assert reported["schema"] == SCHEMA_ID
    assert reported["ok"] is True
    assert reported["problems"] == []
    assert mine == []


def test_catalog_json_is_byte_stable_across_hash_seeds(tmp_path: Path) -> None:
    first = _lab(tmp_path, ["capabilities", "list", "--json"], PYTHONHASHSEED="0")
    second = _lab(tmp_path, ["capabilities", "list", "--json"], PYTHONHASHSEED="7")
    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    executable = _lab(
        tmp_path,
        ["capabilities", "list", "--json", "--executable-only"],
        PYTHONHASHSEED="3",
    )
    assert executable.returncode == 0, executable.stderr
    catalog = CapabilityCatalog.from_json(executable.stdout)
    again = CapabilityCatalog.from_json(executable.stdout)
    assert catalog.digest == again.digest
    assert {cap.id for cap in catalog.executable()} == set(catalog.capabilities)


def test_full_catalog_including_non_executable_entries_parses(tmp_path: Path) -> None:
    """``list --json`` without a filter is the document the client must accept."""
    proc = _lab(tmp_path, ["capabilities", "list", "--json"])
    assert proc.returncode == 0, proc.stderr
    document = json.loads(proc.stdout)
    catalog = CapabilityCatalog.from_json(document)
    assert len(catalog.capabilities) == len(document["capabilities"])
    idle = [cap for cap in catalog.capabilities.values() if not cap.executable]
    assert idle
    assert all(cap.timeout_s == 0 and cap.max_runtime_s == 0 for cap in idle)
    assert all(cap.max_output_chars == 0 for cap in idle)
    assert all(cap.executable_reason.strip() for cap in idle)
    assert any(" " in cap.id for cap in idle)
    assert any(
        arg.name == "run_dir" and arg.kind == "argument"
        for cap in catalog.capabilities.values()
        for arg in cap.args
    )


def test_executable_null_budget_and_odd_id_stay_rejected(tmp_path: Path) -> None:
    proc = _lab(tmp_path, ["capabilities", "list", "--json", "--executable-only"])
    assert proc.returncode == 0, proc.stderr
    document = json.loads(proc.stdout)
    original = next(cap for cap in document["capabilities"] if cap["id"] == "vol-bench")

    def _one(cap: dict[str, object]) -> dict[str, object]:
        return {
            "schema": document["schema"],
            "generated_by": document["generated_by"],
            "capabilities": [cap],
        }

    null_timeout = json.loads(json.dumps(original))
    null_timeout["command"]["timeout_s"] = None
    with pytest.raises(ContractError, match="timeout_s"):
        CapabilityCatalog.from_json(_one(null_timeout))
    null_limit = json.loads(json.dumps(original))
    null_limit["limits"]["max_runtime_s"] = None
    with pytest.raises(ContractError, match="max_runtime_s"):
        CapabilityCatalog.from_json(_one(null_limit))
    odd = json.loads(json.dumps(original))
    odd["id"] = "vol;bench"
    with pytest.raises(ContractError, match="id"):
        CapabilityCatalog.from_json(_one(odd))
    spaced_injection = json.loads(json.dumps(original))
    spaced_injection["id"] = "allocation compare;rm"
    with pytest.raises(ContractError, match="id"):
        CapabilityCatalog.from_json(_one(spaced_injection))


def test_describe_json_parses_through_the_same_client(tmp_path: Path) -> None:
    proc = _lab(tmp_path, ["capabilities", "describe", "vol-bench", "--json"])
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["view"] == "describe"
    assert "error" not in body
    assert body["capability"]["id"] == "vol-bench"
    catalog = CapabilityCatalog.from_json(proc.stdout)
    described = catalog.get("vol-bench")
    assert described is not None and described.executable
    unknown = _lab(tmp_path, ["capabilities", "describe", "not-a-real-capability", "--json"])
    assert unknown.returncode == 1
    unknown_body = json.loads(unknown.stdout)
    assert unknown_body["view"] == "describe"
    assert unknown_body["id"] == "not-a-real-capability"
    assert unknown_body["capability"] is None
    assert unknown_body["error"] == "unknown capability"
    with pytest.raises(ContractError, match="unknown capability"):
        CapabilityCatalog.from_json(unknown.stdout)


def test_cli_defaults_stay_inside_the_overlay_bounds(tmp_path: Path) -> None:
    assert tuple(sorted(_cli_default_violations())) == _CLICK_OUTSIDE_QUALIFIED_WINDOW
    proc = _lab(tmp_path, ["capabilities", "list", "--json", "--executable-only"])
    assert proc.returncode == 0, proc.stderr
    catalog = CapabilityCatalog.from_json(proc.stdout)
    overlay = {str(entry["id"]): entry for entry in CURATED_OVERLAY}
    for cap in catalog.executable():
        entry = overlay[cap.id]
        provided, roots = _plan_inputs(cap, tmp_path / cap.id)
        validated = validate_args(cap, provided, roots=roots)
        argv = build_argv(cap, validated, roots=roots)
        hints = entry["arg_hints"]
        assert isinstance(hints, dict)
        for spec in cap.args:
            if spec.name not in validated:
                continue
            hint = hints.get(spec.name, {})
            if not isinstance(hint, dict):
                continue
            problem = _outside_overlay(validated[spec.name], hint)
            assert problem is None, f"{cap.id} {spec.name}: {problem}"
        for flag, value in _required_pins(entry):
            assert _argv_has_pin(argv, flag, value), f"{cap.id} missing {flag} {value} in {argv}"


def test_qualification_evidence_mentions_the_command() -> None:
    assert _evidence_gaps() == []


def test_verify_receipt_accepts_a_committed_receipt(tmp_path: Path) -> None:
    proc = _lab(tmp_path, ["verify-receipt", str(_RECEIPT)])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["valid"] is True


def test_score_forecasts_seals_a_synthetic_receipt(tmp_path: Path) -> None:
    data = tmp_path / "forecasts.csv"
    rows = [
        f"2020-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00+00:00,1.0,2.0"
        for index in range(30)
    ]
    data.write_text(
        "# SYNTHETIC\ntimestamp,realized,forecast_var\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    proc = _lab(
        tmp_path,
        [
            "score-forecasts",
            "--task",
            "volatility",
            "--data",
            str(data),
            "--out-dir",
            str(out),
            "--receipt-version",
            "2",
        ],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DATA_LABEL=SYNTHETIC" in proc.stdout
    assert "SYNTHETIC" in proc.stdout
    receipt = _receipt_path(proc.stdout)
    checked = _lab(tmp_path, ["verify-receipt", str(receipt)])
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["valid"] is True


def test_vol_bench_smallest_legal_run_verifies(tmp_path: Path) -> None:
    out = tmp_path / "vol"
    proc = _lab(
        tmp_path,
        [
            "vol-bench",
            "--models",
            "rv_roll",
            "--shards",
            "garch_vol",
            "--horizons",
            "1",
            "--min-history",
            "32",
            "--n-origins",
            "10",
            "--stride",
            "1",
            "--n-bars",
            "48",
            "--seed",
            "1",
            "--out-dir",
            str(out),
            "--receipt-version",
            "2",
        ],
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "DATA_LABEL=SYNTHETIC" in proc.stdout
    checked = _lab(tmp_path, ["verify-receipt", str(_receipt_path(proc.stdout))])
    assert checked.returncode == 0, checked.stdout
    assert json.loads(checked.stdout)["valid"] is True


def test_calibration_eval_smallest_legal_run_verifies(tmp_path: Path) -> None:
    out = tmp_path / "cal"
    proc = _lab(
        tmp_path,
        [
            "calibration-eval",
            "--models",
            "gaussian",
            "--shards",
            "iid_gaussian",
            "--n-train",
            "32",
            "--n-eval",
            "64",
            "--pit-bins",
            "5",
            "--seed",
            "1",
            "--out-dir",
            str(out),
            "--receipt-version",
            "2",
        ],
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "DATA_LABEL=SYNTHETIC" in proc.stdout
    checked = _lab(tmp_path, ["verify-receipt", str(_receipt_path(proc.stdout))])
    assert checked.returncode == 0, checked.stdout
    assert json.loads(checked.stdout)["valid"] is True
