"""Path containment, symlink refusal, and private output modes."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import typer

from quant_fund.audit.errors import AuditError
from quant_fund.audit.ledger import AuditLedger
from quant_fund.data.lakehouse.query import _lake_object
from quant_fund.hedge_lab import v2_slate as v2
from quant_fund.monitoring.monitor_chain import MonitorChain
from quant_fund.schemas.errors import DataContractError
from quant_fund.stress.cli import report_cmd
from quant_fund.stress.strategy import strategy_from_mapping
from quant_fund.utils.path_safety import (
    PathSafetyError,
    contained_path,
    open_private,
    output_path,
    write_private_text,
)

_POSITIONS = {"name": "probe", "positions": [{"factor": "us_equity", "weight": 1.0}]}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_contained_path_rejects_escape_and_symlink(tmp_path: Path) -> None:
    base = tmp_path / "strategy"
    base.mkdir()
    inside = contained_path(base, "panel.csv")
    assert inside == (base / "panel.csv").resolve()
    assert inside.is_relative_to(base.resolve())

    with pytest.raises(PathSafetyError, match="escapes"):
        contained_path(base, "../panel.csv")
    with pytest.raises(PathSafetyError, match="escapes"):
        contained_path(base, str(tmp_path / "panel.csv"))
    with pytest.raises(PathSafetyError, match="control"):
        contained_path(base, "panel\n.csv")

    outside = tmp_path / "secret.csv"
    outside.write_text("secret", encoding="utf-8")
    link = base / "panel.csv"
    link.symlink_to(outside)
    with pytest.raises(PathSafetyError, match="symlink"):
        contained_path(base, "panel.csv")
    assert outside.read_text(encoding="utf-8") == "secret"


def test_private_writes_ignore_umask_and_symlinks(tmp_path: Path) -> None:
    previous = os.umask(0)
    try:
        written = write_private_text(tmp_path / "report.md", "body")
        fd = open_private(tmp_path / "chain.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(fd, b"row\n")
        os.close(fd)
    finally:
        os.umask(previous)
    assert written.read_text(encoding="utf-8") == "body"
    assert _mode(written) == 0o600
    assert _mode(tmp_path / "chain.jsonl") == 0o600
    assert (tmp_path / "chain.jsonl").read_text(encoding="utf-8") == "row\n"

    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    link = tmp_path / "via-link.txt"
    link.symlink_to(outside)
    with pytest.raises(PathSafetyError, match="unsafe"):
        write_private_text(link, "pwned")
    with pytest.raises(PathSafetyError, match="unsafe"):
        open_private(link, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_output_path_stays_inside_a_permitted_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifacts" / "lane.json"
    assert output_path(str(artifact), root) == Path(str(artifact))
    with pytest.raises(PathSafetyError, match="escapes"):
        output_path(str(tmp_path / "outside.json"), root)
    with pytest.raises(PathSafetyError, match="escapes"):
        output_path("../outside.json", root)


def test_strategy_returns_csv_cannot_leave_its_directory(tmp_path: Path) -> None:
    base = tmp_path / "strategy"
    base.mkdir()
    loaded = strategy_from_mapping({**_POSITIONS, "returns_csv": "panel.csv"}, base_dir=base)
    assert loaded.returns_csv == str((base / "panel.csv").resolve())

    with pytest.raises(ValueError, match="escapes"):
        strategy_from_mapping({**_POSITIONS, "returns_csv": "../panel.csv"}, base_dir=base)
    with pytest.raises(ValueError, match="escapes"):
        strategy_from_mapping(
            {**_POSITIONS, "returns_csv": str(tmp_path / "panel.csv")}, base_dir=base
        )

    outside = tmp_path / "secret.csv"
    outside.write_text("x", encoding="utf-8")
    (base / "panel.csv").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        strategy_from_mapping({**_POSITIONS, "returns_csv": "panel.csv"}, base_dir=base)


def test_stress_report_is_private_and_refuses_symlink(tmp_path: Path) -> None:
    strategy = tmp_path / "strategy.yaml"
    strategy.write_text(
        "name: probe\nresearch_only: true\npositions:\n  - factor: us_equity\n    weight: 1.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    report_cmd(
        strategy=strategy,
        out=out,
        fmt="markdown",
        returns=None,
        level=0.95,
        seed=0,
        n_scenarios=32,
        n_boot=50,
    )
    assert _mode(out) == 0o600
    sidecar = out.with_suffix(".json")
    assert _mode(sidecar) == 0o600
    assert "research_only" in sidecar.read_text(encoding="utf-8")

    outside = tmp_path / "outside.md"
    outside.write_text("keep", encoding="utf-8")
    link = tmp_path / "linked.md"
    link.symlink_to(outside)
    with pytest.raises(typer.BadParameter, match="unsafe"):
        report_cmd(
            strategy=strategy,
            out=link,
            fmt="markdown",
            returns=None,
            level=0.95,
            seed=0,
            n_scenarios=32,
            n_boot=50,
        )
    assert outside.read_text(encoding="utf-8") == "keep"


def test_monitor_chain_is_private_and_refuses_symlink(tmp_path: Path) -> None:
    path = tmp_path / "ops.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    chain = MonitorChain(path)
    chain.append("gate", {"name": "x"}, ts=1.0)
    assert chain.verify()["ok"] is True
    assert _mode(path) == 0o600

    outside = tmp_path / "outside.jsonl"
    outside.write_text("keep", encoding="utf-8")
    link = tmp_path / "linked.jsonl"
    link.symlink_to(outside)
    with pytest.raises(PathSafetyError, match="symlink"):
        MonitorChain(link).append("gate", {"name": "x"}, ts=1.0)
    verdict = MonitorChain(link).verify()
    assert verdict["ok"] is False
    assert any("symlink" in error for error in verdict["errors"])
    assert outside.read_text(encoding="utf-8") == "keep"


def test_audit_ledger_files_are_private_and_lock_refuses_symlink(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    ledger = AuditLedger(root, sync=False, clock=lambda: "2020-01-01T00:00:00Z")
    ledger.append(
        "research_run",
        {"receipt_sha256": "ab" * 32, "live_pnl_claim": False, "note": "run"},
    )
    entries = root / "entries.jsonl"
    entries.chmod(0o644)
    ledger.append("risk_decision", {"order_id": "o1", "accepted": True, "simulation_only": True})
    assert _mode(entries) == 0o600
    assert _mode(root / ".lock") == 0o600

    escaped = tmp_path / "escaped.jsonl"
    escaped.write_text("keep", encoding="utf-8")
    linked_root = tmp_path / "linked-ledger"
    linked_root.mkdir()
    (linked_root / ".lock").symlink_to(escaped)
    linked = AuditLedger(linked_root, sync=False, clock=lambda: "2020-01-01T00:00:00Z")
    with pytest.raises(AuditError, match="unsafe"):
        linked.append(
            "risk_decision", {"order_id": "o1", "accepted": True, "simulation_only": True}
        )
    assert escaped.read_text(encoding="utf-8") == "keep"
    assert not (linked_root / "entries.jsonl").exists()


def test_receipt_write_stays_inside_root_and_is_private(tmp_path: Path) -> None:
    root = tmp_path / "root"
    artifact = root / "artifacts" / "lane.json"
    written = v2._write_receipt({"a": 1}, str(artifact), root)
    assert written["artifact_path"] == str(artifact)
    assert _mode(artifact) == 0o600
    meta = root / "metadata" / "lane.json"
    assert _mode(meta) == 0o600

    outside = tmp_path / "outside.json"
    outside.write_text("keep", encoding="utf-8")
    with pytest.raises(PathSafetyError, match="escapes"):
        v2._write_receipt({"a": 1}, str(outside), root)
    with pytest.raises(PathSafetyError, match="escapes"):
        v2._write_receipt({"a": 1}, "../lane.json", root)
    link = root / "linked.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    with pytest.raises(PathSafetyError, match="escapes"):
        v2._write_receipt({"a": 1}, str(link), root)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_lane_receipt_and_lake_object_reject_escape(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    receipt = root / "ml_lane.json"
    receipt.write_text("{}", encoding="utf-8")
    assert v2._contained_lane_receipt(receipt, root.resolve()) == receipt.resolve()

    outside = tmp_path / "other.json"
    outside.write_text("{}", encoding="utf-8")
    link = root / "linked.json"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        v2._contained_lane_receipt(link, root.resolve())
    with pytest.raises(ValueError, match="escapes"):
        v2._contained_lane_receipt(outside, root.resolve())

    lake = tmp_path / "lake"
    (lake / "objects").mkdir(parents=True)
    object_link = lake / "objects" / "obj.parquet"
    object_link.symlink_to(outside)
    with pytest.raises(DataContractError, match="escapes"):
        _lake_object(lake, "objects/obj.parquet")
    with pytest.raises(DataContractError, match="escapes"):
        _lake_object(lake, "../other.json")
    safe = _lake_object(lake, "objects/missing.parquet")
    assert safe.is_relative_to(lake.resolve())
    assert outside.read_text(encoding="utf-8") == "{}"
