"""CLI bridge for the allocation research pack.

``quant_fund.research.allocation`` shipped fully documented and tested with no
operator surface — no CLI, no script, no API route. These tests drive the new
``dipcatcher allocation`` sub-typer end-to-end on seeded SYNTHETIC panels and
pin both the honesty contract and the construction science (risk parity must
actually equalise risk contributions; the vol-target engine must refuse to run
without a target).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from typer.testing import CliRunner

from quant_fund.research.allocation import ENGINE_NAMES
from quant_fund.research.allocation_cli import _data_label, allocation_app
from quant_fund.research.catalog.registry import FORBIDDEN_RESEARCH_METRIC_KEYS

runner = CliRunner()

_SMALL = ["--n-assets", "4", "--n-periods", "240", "--window", "60", "--step", "10"]


def test_data_label_matches_the_canonical_cli_formatter() -> None:
    """The local copy may not drift from ``cli.support.format_data_label``.

    Duplicated because the ``library-no-cli-imports`` architecture guard
    forbids a research-layer module from importing the CLI, even lazily.
    """
    from quant_fund.cli.support import format_data_label

    assert _data_label(synthetic=True, data_source="ignored") == format_data_label(
        synthetic=True, data_source="ignored"
    )
    assert _data_label(synthetic=False, data_source="bars.csv") == format_data_label(
        synthetic=False, data_source="bars.csv"
    )


def test_engines_lists_the_registry() -> None:
    result = runner.invoke(allocation_app, ["engines"])
    assert result.exit_code == 0, result.output
    listed = [line for line in result.output.splitlines() if line.strip()]
    assert listed == list(ENGINE_NAMES)


def test_walk_forward_runs_and_seals_a_v1_receipt(tmp_path: Path) -> None:
    out = tmp_path / "alloc.json"
    result = runner.invoke(
        allocation_app,
        ["walk-forward", "--engine", "risk_parity", *_SMALL, "--seed", "1", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "constraint_violations=0" in result.output
    assert f"receipt={out}" in result.output

    lowered = result.output.lower()
    for token in FORBIDDEN_RESEARCH_METRIC_KEYS:
        assert token not in lowered, f"headlined forbidden metric {token!r}"

    from quant_fund.research.allocation import verify_allocation_receipt

    verification = verify_allocation_receipt(out)
    assert verification["valid"], verification["errors"]
    receipt = json.loads(out.read_text())
    assert receipt["schema"] == "allocation_evaluation.v1"
    assert receipt["synthetic"] is True
    assert receipt["claim"] == "research_only"
    assert receipt["engine"] == "risk_parity"


def test_walk_forward_v2_receipt_seals_and_verifies(tmp_path: Path) -> None:
    out = tmp_path / "alloc_v2.json"
    result = runner.invoke(
        allocation_app,
        [
            "walk-forward",
            "--engine",
            "inverse_volatility",
            *_SMALL,
            "--seed",
            "1",
            "--out",
            str(out),
            "--receipt-version",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    sealed = json.loads(out.read_text())
    assert sealed["schema"] == "receipt.v2"
    assert sealed["receipt_sha256"]
    assert sealed["live_pnl_claim"] is False
    assert sealed["data_label"] == "SYNTHETIC"

    from quant_fund.research.allocation import verify_allocation_receipt

    assert verify_allocation_receipt(out)["valid"]


def test_walk_forward_rejects_unknown_engine(tmp_path: Path) -> None:
    result = runner.invoke(
        allocation_app,
        ["walk-forward", "--engine", "not_an_engine", *_SMALL, "--out", str(tmp_path / "x.json")],
    )
    assert result.exit_code != 0
    assert "unknown engine" in result.output
    assert ", ".join(ENGINE_NAMES) in result.output  # remediation lists the registry


def test_walk_forward_vol_target_needs_a_target(tmp_path: Path) -> None:
    result = runner.invoke(
        allocation_app,
        ["walk-forward", "--engine", "vol_target", *_SMALL, "--out", str(tmp_path / "x.json")],
    )
    assert result.exit_code != 0
    assert "--target-vol" in result.output


def test_walk_forward_bad_receipt_version(tmp_path: Path) -> None:
    result = runner.invoke(
        allocation_app,
        [
            "walk-forward",
            "--engine",
            "risk_parity",
            *_SMALL,
            "--out",
            str(tmp_path / "x.json"),
            "--receipt-version",
            "3",
        ],
    )
    assert result.exit_code != 0
    assert "--receipt-version must be 1 or 2" in result.output


def test_compare_default_set_excludes_vol_target() -> None:
    """A bare ``allocation compare`` must run: vol_target needs a target."""
    result = runner.invoke(allocation_app, ["compare", *_SMALL, "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    for name in ENGINE_NAMES:
        if name == "vol_target":
            assert name not in result.output
        else:
            assert name in result.output
    assert "lowest_vol_rmse=" in result.output
    # The ranking is in-sample on one panel; the command must say so.
    assert "not an out-of-sample claim" in result.output


def test_compare_with_target_vol_includes_vol_target(tmp_path: Path) -> None:
    result = runner.invoke(
        allocation_app, ["compare", *_SMALL, "--seed", "1", "--target-vol", "0.10"]
    )
    assert result.exit_code == 0, result.output
    assert "vol_target" in result.output
    assert "target_vol_rmse=" in result.output


def test_compare_risk_parity_equalises_risk_contributions() -> None:
    """The construction science: risk parity drives the ERC residual to ~0,
    and strictly below an inverse-volatility book on the same panel."""
    result = runner.invoke(allocation_app, ["compare", *_SMALL, "--seed", "1"])
    assert result.exit_code == 0, result.output
    erc: dict[str, float] = {}
    for line in result.output.splitlines():
        parts = line.split()
        if not parts or "erc_res_mean=" not in line:
            continue
        token = next(p for p in parts if p.startswith("erc_res_mean="))
        erc[parts[0]] = float(token.split("=", 1)[1])
    assert "risk_parity" in erc and "inverse_volatility" in erc
    assert erc["risk_parity"] < 1e-6, erc
    assert erc["risk_parity"] < erc["inverse_volatility"], erc


def test_compare_writes_one_receipt_per_engine(tmp_path: Path) -> None:
    out = tmp_path / "receipts"
    result = runner.invoke(
        allocation_app, ["compare", *_SMALL, "--seed", "1", "--out-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in out.glob("allocation_*.json"))
    expected = sorted(f"allocation_{name}.json" for name in ENGINE_NAMES if name != "vol_target")
    assert written == expected

    from quant_fund.research.allocation import verify_allocation_receipt

    for path in out.glob("*.json"):
        assert verify_allocation_receipt(path)["valid"], path


def test_returns_file_relabels_the_evidence(tmp_path: Path) -> None:
    """--returns switches the label off SYNTHETIC; it does not make it proof."""
    rng = np.random.default_rng(7)
    frame = pl.DataFrame(
        {f"a{i}": rng.normal(0.0, 0.01, 200) for i in range(3)},
    )
    csv = tmp_path / "returns.csv"
    frame.write_csv(csv)

    out = tmp_path / "alloc.json"
    result = runner.invoke(
        allocation_app,
        [
            "walk-forward",
            "--engine",
            "risk_parity",
            "--returns",
            str(csv),
            "--window",
            "60",
            "--step",
            "10",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=returns.csv" in result.output
    assert "DATA_LABEL=SYNTHETIC" not in result.output
    receipt = json.loads(out.read_text())
    assert receipt["synthetic"] is False
    assert receipt["parameters"]["returns_file"] == str(csv)
    assert receipt["parameters"]["seed"] is None


def test_verify_receipt_exits_nonzero_on_tamper(tmp_path: Path) -> None:
    out = tmp_path / "alloc.json"
    result = runner.invoke(
        allocation_app,
        ["walk-forward", "--engine", "risk_parity", *_SMALL, "--seed", "1", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output

    clean = runner.invoke(allocation_app, ["verify-receipt", str(out)])
    assert clean.exit_code == 0, clean.output
    assert '"valid": true' in clean.output

    payload = json.loads(out.read_text())
    payload["metrics"]["vol_rmse"] = 0.0
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    tampered = runner.invoke(allocation_app, ["verify-receipt", str(out)])
    assert tampered.exit_code == 1
    assert '"valid": false' in tampered.output
    assert "receipt_payload_hash_mismatch" in tampered.output


def test_synthetic_panel_rejects_degenerate_sizes(tmp_path: Path) -> None:
    result = runner.invoke(
        allocation_app,
        ["walk-forward", "--engine", "risk_parity", "--n-assets", "1", "--n-periods", "100"],
    )
    assert result.exit_code != 0
    assert "--n-assets must be >= 2" in result.output
