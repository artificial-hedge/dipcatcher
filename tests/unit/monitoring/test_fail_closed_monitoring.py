"""Fail-closed error handling in ``quant_fund.monitoring``.

The ops layer's contract is that a missing, unreadable, or contradictory
input is never reported as healthy. These tests pin the failure *classes*
that must not silently pass:

- ``drift.verify_evidence_report_sidecar`` — the ``reason`` must distinguish
  a missing report, a missing sidecar, an unreadable sidecar, a hash
  mismatch, and a corrupt-but-hash-matching report. Collapsing them into
  ``hash_mismatch`` sends an operator chasing the wrong defect.
- ``monitor_chain`` — a corrupt, non-object, or non-UTF-8 ledger line must
  fail closed instead of being read as an empty or short chain.
- ``dashboard.ops_snapshot`` — the aggregate verdict must not read ``ok``
  while any individual check is ``insufficient_data``, and a config missing
  its risk-gate limits must be named rather than defaulted.
- ``kill_switch`` — an unknown state blocks every action.

Every fixture is a ``tmp_path`` file. No credential path, receipt store, or
real data directory is read or written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.config.models import AppConfig, KillSwitchConfig
from quant_fund.monitoring.dashboard import ops_snapshot, render_markdown
from quant_fund.monitoring.drift import verify_evidence_report_sidecar
from quant_fund.monitoring.kill_switch import (
    CANCEL_OPEN_ORDERS,
    ENABLED,
    FLATTEN_OPTIONAL,
    HALT_NEW_ORDERS,
    KillSwitch,
)
from quant_fund.monitoring.monitor_chain import KINDS, MonitorChain
from quant_fund.schemas.errors import KillSwitchActive

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_report(
    directory: Path,
    *,
    payload: bytes = b'{"status": "complete", "warnings": []}',
    sidecar: bytes | None = None,
    name: str = "evidence_report.json",
) -> Path:
    """Write a report and a sidecar digest that matches ``payload``."""
    path = directory / name
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest() if sidecar is None else sidecar.decode()
    (directory / f"{name}.sha256").write_text(digest + "\n", encoding="ascii")
    return path


def _cfg(**risk: Any) -> AppConfig:
    config = AppConfig()
    for key, value in risk.items():
        setattr(config.risk_gate, key, value)
    return config


def _full_snapshot(config: AppConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """A snapshot with every optional input supplied — nothing insufficient."""
    payload: dict[str, Any] = {
        "nav": 1_000_000.0,
        "cash": 1_000_000.0,
        "positions": {},
        "marks": {},
        "config": config if config is not None else AppConfig(),
        "mark_age_bars": {},
        "n_open_orders": 0,
        "kill_switch_state": "ENABLED",
        "peak_nav": 1_100_000.0,
        "recon_mismatches": 0,
        "drift_alert": False,
    }
    payload.update(overrides)
    return ops_snapshot(**payload)


# --------------------------------------------------------------------------- #
# drift.verify_evidence_report_sidecar — failure reasons
# --------------------------------------------------------------------------- #


def test_sidecar_missing_report_is_invalid(tmp_path: Path) -> None:
    verdict = verify_evidence_report_sidecar(tmp_path / "absent.json")
    assert verdict["valid"] is False
    assert verdict["status"] == "invalid"
    assert verdict["reason"] == "missing"


def test_sidecar_missing_sidecar_is_invalid(tmp_path: Path) -> None:
    report = tmp_path / "evidence_report.json"
    report.write_text('{"status": "complete"}', encoding="utf-8")
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is False
    assert verdict["reason"] == "missing_sidecar"


def test_sidecar_hash_mismatch_is_invalid(tmp_path: Path) -> None:
    report = _write_report(tmp_path, sidecar=b"0" * 64)
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is False
    assert verdict["reason"] == "hash_mismatch"


def test_sidecar_corrupt_report_with_matching_digest_is_corrupt_not_mismatch(
    tmp_path: Path,
) -> None:
    """A report whose bytes hash correctly but do not parse is ``corrupt_report``.

    The digest matched, so labelling this ``hash_mismatch`` would point an
    operator at the wrong defect.
    """
    report = _write_report(tmp_path, payload=b"{not json at all")
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is False
    assert verdict["reason"] == "corrupt_report"


def test_sidecar_non_utf8_report_is_corrupt(tmp_path: Path) -> None:
    report = _write_report(tmp_path, payload=b"\xff\xfe\x00binary")
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is False
    assert verdict["reason"] == "corrupt_report"


def test_sidecar_non_ascii_sidecar_is_unreadable(tmp_path: Path) -> None:
    report = tmp_path / "evidence_report.json"
    report.write_text('{"status": "complete"}', encoding="utf-8")
    (tmp_path / "evidence_report.json.sha256").write_bytes("café\n".encode())
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is False
    assert verdict["reason"] == "unreadable_sidecar"


def test_sidecar_valid_report_is_verified(tmp_path: Path) -> None:
    report = _write_report(tmp_path)
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is True
    assert verdict["reason"] == "verified"
    assert verdict["live_pnl_claim"] is False
    assert verdict["research_only"] is True


def test_sidecar_valid_report_carries_warnings(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        payload=json.dumps({"status": "insufficient_evidence", "warnings": ["x"]}).encode(),
    )
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is True
    assert verdict["status"] == "insufficient_evidence"
    assert verdict["warnings"] == ["x"]


def test_sidecar_report_that_is_a_json_list_is_insufficient_not_valid(tmp_path: Path) -> None:
    report = _write_report(tmp_path, payload=b"[1, 2, 3]")
    verdict = verify_evidence_report_sidecar(report)
    assert verdict["valid"] is True
    assert verdict["status"] == "insufficient_evidence"
    assert verdict["warnings"] == []


def test_sidecar_directory_instead_of_file_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "evidence_report.json").mkdir()
    verdict = verify_evidence_report_sidecar(tmp_path / "evidence_report.json")
    assert verdict["valid"] is False
    assert verdict["reason"] == "missing"


# --------------------------------------------------------------------------- #
# monitor_chain — corrupt ledger files fail closed
# --------------------------------------------------------------------------- #


def test_chain_verify_is_ok_after_appends(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.append("dashboard", {"status": "ok"})
    chain.append("drift", {"status": "ok"})
    verdict = chain.verify()
    assert verdict["ok"] is True
    assert verdict["errors"] == []
    assert verdict["n"] == 2


def test_chain_empty_file_verifies(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.path.write_text("", encoding="utf-8")
    verdict = chain.verify()
    assert verdict["ok"] is True
    assert verdict["n"] == 0
    assert verdict["tip"] == "0" * 64


def test_chain_corrupt_line_fails_closed(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.append("dashboard", {"status": "ok"})
    with chain.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    verdict = chain.verify()
    assert verdict["ok"] is False
    assert verdict["errors"]
    assert "not JSON" in verdict["errors"][0]


def test_chain_non_object_line_fails_closed(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.append("dashboard", {"status": "ok"})
    with chain.path.open("a", encoding="utf-8") as handle:
        handle.write("[1, 2, 3]\n")
    verdict = chain.verify()
    assert verdict["ok"] is False
    assert "not an object" in verdict["errors"][0]


def test_chain_non_utf8_file_fails_closed(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.path.write_bytes(b"\xff\xfe\x00binary")
    verdict = chain.verify()
    assert verdict["ok"] is False
    assert "not valid UTF-8" in verdict["errors"][0]


def test_chain_refuses_to_append_to_a_corrupt_ledger(tmp_path: Path) -> None:
    """A corrupt ledger must not be extended — that would launder the defect."""
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        chain.append("dashboard", {"status": "ok"})


def test_chain_refuses_to_append_to_a_non_utf8_ledger(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.path.write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        chain.append("dashboard", {"status": "ok"})


def test_chain_detects_a_deleted_record(tmp_path: Path) -> None:
    """Removing a record from the middle of the ledger breaks the chain."""
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.append("dashboard", {"n": 1})
    chain.append("dashboard", {"n": 2})
    chain.append("dashboard", {"n": 3})
    lines = chain.path.read_text(encoding="utf-8").splitlines()
    chain.path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
    verdict = chain.verify()
    assert verdict["ok"] is False
    assert any("seq_gap" in error for error in verdict["errors"])
    assert any("chain_break" in error for error in verdict["errors"])


def test_chain_cannot_detect_a_truncated_tail(tmp_path: Path) -> None:
    """Documented limitation: a hash chain has no external anchor.

    Dropping the *newest* record leaves a self-consistent prefix, so the
    chain still verifies. Detecting that needs a signed head kept outside
    the file — the chain itself cannot answer it.
    """
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.append("dashboard", {"n": 1})
    chain.append("dashboard", {"n": 2})
    lines = chain.path.read_text(encoding="utf-8").splitlines()
    chain.path.write_text(lines[0] + "\n", encoding="utf-8")
    verdict = chain.verify()
    assert verdict["ok"] is True
    assert verdict["n"] == 1


def test_chain_detects_a_rewritten_record(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    chain.append("dashboard", {"n": 1})
    rec = json.loads(chain.path.read_text(encoding="utf-8").splitlines()[0])
    rec["detail"]["n"] = 99
    chain.path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    verdict = chain.verify()
    assert verdict["ok"] is False
    assert any("content_edit" in error for error in verdict["errors"])


def test_chain_rejects_an_unknown_kind(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    with pytest.raises(ValueError, match="kind must be one of"):
        chain.append("not-a-kind", {})
    assert "not-a-kind" not in KINDS


def test_chain_tip_of_an_empty_chain_is_genesis(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    assert chain.tip() == "0" * 64


def test_chain_tip_matches_the_newest_record(tmp_path: Path) -> None:
    chain = MonitorChain(tmp_path / "chain.jsonl")
    record = chain.append("dashboard", {"status": "ok"})
    assert chain.tip() == record["sha256"]


# --------------------------------------------------------------------------- #
# dashboard.ops_snapshot — the aggregate verdict fails closed
# --------------------------------------------------------------------------- #


def test_snapshot_with_every_input_is_ok() -> None:
    snap = _full_snapshot()
    assert snap["overall_status"] == "ok"
    assert all(check["status"] == "ok" for check in snap["checks"].values())


def test_snapshot_missing_inputs_is_not_ok() -> None:
    """The aggregate may not read ``ok`` while a check is ``insufficient_data``."""
    snap = ops_snapshot(
        nav=1_000_000.0,
        cash=1_000_000.0,
        positions={},
        marks={},
        config=AppConfig(),
    )
    assert snap["overall_status"] == "insufficient_data"
    assert snap["checks"]["kill_switch"]["status"] == "insufficient_data"
    assert snap["checks"]["reconciliation"]["status"] == "insufficient_data"
    assert snap["checks"]["feature_drift"]["status"] == "insufficient_data"


def test_snapshot_breach_outranks_insufficient_data() -> None:
    snap = _full_snapshot(kill_switch_state="HALT_NEW_ORDERS")
    assert snap["checks"]["kill_switch"]["status"] == "breach"
    assert snap["overall_status"] == "breach"


def test_snapshot_warn_outranks_insufficient_data() -> None:
    snap = _full_snapshot(cash=-1.0)
    assert snap["checks"]["cash_buffer"]["status"] == "warn"
    assert snap["overall_status"] == "warn"


def test_snapshot_markdown_shows_the_insufficient_overall() -> None:
    snap = ops_snapshot(
        nav=1_000_000.0,
        cash=1_000_000.0,
        positions={},
        marks={},
        config=AppConfig(),
    )
    markdown = render_markdown(snap)
    assert "- overall: **insufficient_data**" in markdown


def test_snapshot_markdown_shows_the_ok_overall() -> None:
    markdown = render_markdown(_full_snapshot())
    assert "- overall: **ok**" in markdown


def test_snapshot_rejects_a_non_finite_nav() -> None:
    with pytest.raises(ValueError, match="nav"):
        ops_snapshot(
            nav=float("nan"),
            cash=0.0,
            positions={},
            marks={},
            config=AppConfig(),
        )


def test_snapshot_rejects_a_non_positive_nav() -> None:
    with pytest.raises(ValueError, match="nav"):
        ops_snapshot(
            nav=0.0,
            cash=0.0,
            positions={},
            marks={},
            config=AppConfig(),
        )


def test_snapshot_names_the_missing_risk_gate_limits() -> None:
    """A missing config key is named, never defaulted to "no limit"."""

    class PartialRiskGate:
        max_gross = 2.0

    class PartialConfig:
        risk_gate = PartialRiskGate()

    with pytest.raises(ValueError, match="max_net"):
        ops_snapshot(
            nav=1.0,
            cash=1.0,
            positions={},
            marks={},
            config=PartialConfig(),  # type: ignore[arg-type]
        )


def test_snapshot_requires_a_risk_gate() -> None:
    class NoRiskGate:
        pass

    with pytest.raises(ValueError, match="config.risk_gate"):
        ops_snapshot(
            nav=1.0,
            cash=1.0,
            positions={},
            marks={},
            config=NoRiskGate(),  # type: ignore[arg-type]
        )


def test_snapshot_unmarked_position_is_a_breach() -> None:
    snap = _full_snapshot(positions={"BTC": 5.0}, marks={})
    assert snap["checks"]["unmarked_positions"]["status"] == "breach"
    assert snap["checks"]["unmarked_positions"]["securities"] == ["BTC"]
    assert snap["overall_status"] == "breach"


def test_snapshot_stale_mark_is_a_breach() -> None:
    snap = _full_snapshot(mark_age_bars={"BTC": 99})
    assert snap["checks"]["mark_staleness"]["status"] == "breach"
    assert snap["checks"]["mark_staleness"]["stale_securities"] == ["BTC"]


def test_snapshot_reconciliation_mismatch_is_a_breach() -> None:
    snap = _full_snapshot(recon_mismatches=2)
    assert snap["checks"]["reconciliation"]["status"] == "breach"
    assert snap["overall_status"] == "breach"


def test_snapshot_drift_alert_is_a_breach() -> None:
    snap = _full_snapshot(drift_alert=True)
    assert snap["checks"]["feature_drift"]["status"] == "breach"
    assert snap["overall_status"] == "breach"


def test_snapshot_is_never_a_live_pnl_claim() -> None:
    snap = _full_snapshot()
    assert snap["research_only"] is True
    assert snap["live_pnl_claim"] is False


def test_snapshot_uses_the_configured_limits() -> None:
    config = _cfg(max_gross=0.1)
    snap = _full_snapshot(
        config=config, nav=1_000_000.0, cash=0.0, positions={"BTC": 5.0}, marks={"BTC": 60_000.0}
    )
    assert snap["checks"]["gross"]["status"] == "breach"
    assert snap["checks"]["gross"]["limit"] == 0.1


# --------------------------------------------------------------------------- #
# kill_switch — unknown states fail closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("state", ["BOGUS", "", "enabled", "HALT", "0", "None"])
def test_unknown_kill_switch_state_blocks_orders(state: str) -> None:
    switch = KillSwitch(KillSwitchConfig(state=state))
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()


@pytest.mark.parametrize("state", ["BOGUS", "", "enabled", "0"])
def test_unknown_kill_switch_state_denies_cancel_and_flatten(state: str) -> None:
    switch = KillSwitch(KillSwitchConfig(state=state))
    assert switch.may_cancel_open_orders() is False
    assert switch.may_flatten(True) is False


def test_kill_switch_set_state_rejects_unknown_names() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    with pytest.raises(KillSwitchActive):
        switch.set_state("NOT_A_STATE")
    assert switch.state == ENABLED


def test_kill_switch_enabled_allows_orders() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    switch.assert_new_orders_allowed()
    assert switch.may_cancel_open_orders() is False
    assert switch.may_flatten(True) is False


def test_kill_switch_flatten_requires_human_authorization() -> None:
    switch = KillSwitch(KillSwitchConfig(state=FLATTEN_OPTIONAL))
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()
    assert switch.may_cancel_open_orders() is True
    assert switch.may_flatten(False) is False
    assert switch.may_flatten(True) is True


def test_kill_switch_auto_flatten_never_flattens() -> None:
    switch = KillSwitch(KillSwitchConfig(state=FLATTEN_OPTIONAL, allow_auto_flatten=True))
    assert switch.may_flatten(True) is False


@pytest.mark.parametrize("state", [HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL])
def test_kill_switch_blocking_states_block_orders(state: str) -> None:
    switch = KillSwitch(KillSwitchConfig(state=state))
    with pytest.raises(KillSwitchActive):
        switch.assert_new_orders_allowed()
