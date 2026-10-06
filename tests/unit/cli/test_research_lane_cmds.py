"""CLI bridge for the Wave-1 research lanes (coherence / concordance /
multih-fleet / expert-mixture / compare).

These lanes already produced sealed evidence — ``receipts/`` carries committed
``coherence_eval``, ``selection_concordance`` and ``multih_fleet_eval``
receipts, and ``receipt_v2`` had lane contracts for three of them — but nothing
in the CLI could produce one. The only callers were tests and hand-run scripts.

Every test here drives the real implementation end-to-end on seeded SYNTHETIC
input (no monkeypatching of the lane itself) and asserts the honesty contract:
``DATA_LABEL=`` is always printed, the receipt seals and re-verifies, and no
forbidden headline metric leaks into the output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.catalog.registry import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.research.receipt_v2 import verify_receipt_file

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Normalise typer's rich error box so a wrapped message is matchable.

    Rich draws usage errors inside a bordered box and wraps the message at the
    terminal width, so a phrase like "pass --dev to acknowledge" is split
    across lines and padded with box characters. Strip ANSI, drop the box
    rules, and collapse whitespace to recover the sentence.
    """
    stripped = _ANSI.sub("", text)
    stripped = stripped.replace("│", " ").replace("╭", " ").replace("╰", " ").replace("─", " ")
    return " ".join(stripped.split())


# Deliberately tiny: these lanes fit a fleet head per shard, so the defaults
# (n_train=512, n_boot=500) would put this file in the slow lane.
_HEADS = "empirical,gaussian,skew_t"
_SHARDS = "iid_gaussian"
_LANE_ARGS = {
    "coherence": ["--panels", "gauss_factor", "--n-train", "96", "--n-eval", "32", "--n-mc", "64"],
    "concordance": [
        "--heads",
        _HEADS,
        "--shards",
        _SHARDS,
        "--n-train",
        "160",
        "--n-eval",
        "64",
        "--n-boot",
        "60",
    ],
    "multih-fleet": [
        "--models",
        "empirical,hstep_t",
        "--shards",
        _SHARDS,
        "--horizons",
        "1,5",
        "--n-train",
        "150",
        "--n-eval",
        "20",
        "--n",
        "280",
    ],
    "expert-mixture": [
        "--dev",
        "--heads",
        _HEADS,
        "--shards",
        _SHARDS,
        "--n-train",
        "160",
        "--n-eval",
        "64",
    ],
}

#: receipt filename prefix per lane, so a test can find what it produced.
_RECEIPT_PREFIX = {
    "coherence": "coherence_",
    "concordance": "concordance_",
    "multih-fleet": "multih_fleet_eval_",
    "expert-mixture": "expert_mixture_",
}

_EXPECTED_KIND = {
    "coherence": "coherence_eval",
    "concordance": "selection_concordance",
    "multih-fleet": "multih_fleet_eval",
    "expert-mixture": "expert_mixture_eval",
}


@pytest.mark.parametrize("cmd", sorted(_LANE_ARGS))
def test_lane_help(cmd: str) -> None:
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, result.output
    assert "--out-dir" in result.output


@pytest.mark.parametrize("cmd", sorted(_LANE_ARGS))
def test_lane_runs_seals_and_reverifies(cmd: str, tmp_path: Path) -> None:
    """Real end-to-end: run the lane, seal a receipt, verify it independently."""
    out = tmp_path / "receipts"
    # ``receipts/`` is immutable evidence (AGENTS.md rule 4) and the lanes
    # default --out-dir to it, so pin that the flag really redirects the write.
    committed = Path("receipts")
    before = sorted(p.name for p in committed.glob("*.json")) if committed.is_dir() else []

    result = runner.invoke(
        app,
        [cmd, *_LANE_ARGS[cmd], "--seed", "5", "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.output

    after = sorted(p.name for p in committed.glob("*.json")) if committed.is_dir() else []
    assert before == after, f"{cmd} wrote into the committed evidence store"

    # Honesty contract: the DATA_LABEL line is always printed and says SYNTHETIC.
    assert "DATA_LABEL=SYNTHETIC" in result.output

    # No forbidden headline metric may appear anywhere in the output.
    lowered = result.output.lower()
    for token in FORBIDDEN_RESEARCH_METRIC_KEYS:
        assert token not in lowered, f"{cmd} headlined forbidden metric {token!r}"

    produced = sorted(out.glob(f"{_RECEIPT_PREFIX[cmd]}*.json"))
    assert len(produced) == 1, f"expected exactly one receipt, got {produced}"
    assert f"receipt={produced[0]}" in result.output
    assert "verdict=pass" in result.output

    verification = verify_receipt_file(produced[0])
    assert verification["valid"], verification["errors"]
    assert verification["kind"] == _EXPECTED_KIND[cmd]

    sealed = json.loads(produced[0].read_text())
    assert sealed["data_label"] == "SYNTHETIC"
    assert sealed["live_pnl_claim"] is False


def test_coherence_rejects_unknown_panel(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["coherence", "--panels", "gauss_factor,not_a_panel", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    plain = _plain(result.output)
    assert "unknown panel" in plain
    assert "gauss_factor" in plain  # remediation lists the registry
    assert not list(tmp_path.glob("*.json"))


def test_coherence_rejects_unknown_method(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["coherence", "--methods", "direct,bogus", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "unknown method" in _plain(result.output)
    assert not list(tmp_path.glob("*.json"))


def test_multih_fleet_rejects_non_integer_horizons(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "multih-fleet",
            "--models",
            "empirical",
            "--shards",
            _SHARDS,
            "--horizons",
            "1,x",
            "--n-train",
            "60",
            "--n-eval",
            "10",
            "--n",
            "90",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "--horizons" in result.output


def test_multih_fleet_rejects_zero_horizon(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "multih-fleet",
            "--models",
            "empirical",
            "--shards",
            _SHARDS,
            "--horizons",
            "0",
            "--n-train",
            "60",
            "--n-eval",
            "10",
            "--n",
            "90",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "at least one horizon >= 1" in _plain(result.output)


def test_concordance_rejects_unknown_head(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "concordance",
            "--heads",
            "empirical,not_a_head",
            "--shards",
            _SHARDS,
            "--n-train",
            "64",
            "--n-eval",
            "32",
            "--n-boot",
            "20",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "not_a_head" in result.output
    assert not list(tmp_path.glob("*.json"))


def test_expert_mixture_requires_dev_flag(tmp_path: Path) -> None:
    """Dev-only lane: no --dev is a typed BadParameter, never a silent run."""
    result = runner.invoke(
        app,
        [
            "expert-mixture",
            "--heads",
            "empirical",
            "--shards",
            _SHARDS,
            "--n-train",
            "64",
            "--n-eval",
            "32",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "pass --dev to acknowledge" in _plain(result.output)
    assert not list(tmp_path.glob("*.json"))


def test_expert_mixture_rejects_unknown_shard(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "expert-mixture",
            "--dev",
            "--heads",
            "empirical",
            "--shards",
            "not_a_shard",
            "--n-train",
            "64",
            "--n-eval",
            "32",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "not_a_shard" in result.output


@pytest.fixture(scope="module")
def two_coherence_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two same-lane runs with different seeds, for the compare lane."""
    base = tmp_path_factory.mktemp("compare")
    paths: list[Path] = []
    for seed in (3, 11):
        out = base / f"run{seed}"
        result = runner.invoke(
            app,
            [
                "coherence",
                "--panels",
                "gauss_factor",
                "--n-train",
                "96",
                "--n-eval",
                "32",
                "--n-mc",
                "64",
                "--seed",
                str(seed),
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        produced = sorted(out.glob("coherence_*.json"))
        assert len(produced) == 1
        paths.append(produced[0])
    return paths[0], paths[1]


def test_compare_markdown(two_coherence_runs: tuple[Path, Path]) -> None:
    run_a, run_b = two_coherence_runs
    result = runner.invoke(app, ["compare", str(run_a), str(run_b), "--n-boot", "100"])
    assert result.exit_code == 0, result.output
    assert "# Run comparison" in result.output
    # A comparison of two stored runs cannot know either was SYNTHETIC.
    assert "DATA_LABEL=UNKNOWN" in result.output
    assert "series=" in result.output


def test_compare_json_and_v2_receipt(two_coherence_runs: tuple[Path, Path], tmp_path: Path) -> None:
    run_a, run_b = two_coherence_runs
    report = tmp_path / "report.json"
    receipt = tmp_path / "cmp.json"
    result = runner.invoke(
        app,
        [
            "compare",
            str(run_a),
            str(run_b),
            "--format",
            "json",
            "--n-boot",
            "100",
            "--out",
            str(report),
            "--receipt-out",
            str(receipt),
            "--receipt-version",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert report.is_file()
    assert json.loads(report.read_text())["run_a"]
    assert receipt.is_file()
    sealed = json.loads(receipt.read_text())
    assert sealed["receipt_sha256"]
    assert sealed["data_label"] == "UNKNOWN"
    assert sealed["live_pnl_claim"] is False
    verification = verify_receipt_file(receipt)
    assert verification["valid"], verification["errors"]


def test_compare_rejects_bad_format(two_coherence_runs: tuple[Path, Path]) -> None:
    run_a, run_b = two_coherence_runs
    result = runner.invoke(app, ["compare", str(run_a), str(run_b), "--format", "yaml"])
    assert result.exit_code != 0
    assert "markdown or json" in _plain(result.output)


def test_compare_rejects_bad_receipt_version(two_coherence_runs: tuple[Path, Path]) -> None:
    run_a, run_b = two_coherence_runs
    result = runner.invoke(
        app,
        ["compare", str(run_a), str(run_b), "--receipt-version", "3", "--receipt-out", "x.json"],
    )
    assert result.exit_code != 0
    assert "--receipt-version must be 1 or 2" in _plain(result.output)


def test_compare_requires_existing_runs(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    result = runner.invoke(app, ["compare", str(missing), str(missing)])
    assert result.exit_code != 0
