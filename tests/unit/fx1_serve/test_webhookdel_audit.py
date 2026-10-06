"""Tests for fx1.serve.webhookdel_audit — the delivery-dispatcher contract lane."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fx1.serve.webhookdel_audit import webhookdel_audit, webhookdel_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

_REPO = Path(__file__).resolve().parents[3]
_RECEIPT = _REPO / "receipts" / "fx1_webhookdel_audit.json"
_MODULE = "src/fx1/serve/webhookdel_audit.py"

# The exact battery contract: a dropped or renamed probe fails this test,
# and an empty results map can never read as ok.
_EXPECTED_PROBES = frozenset(
    {
        "backoff_exponential_doubling",
        "backoff_fresh_connection_each_attempt",
        "backoff_payload_identical_each_attempt",
        "backoff_sig_verified_each_attempt",
        "backoff_total_bounded",
        "backoff_zero_no_sleep",
        "cancel_mid_delivery_409",
        "cancel_mid_delivery_completes",
        "dns_reresolved_per_attempt",
        "drain_latch_leaves_queued_pending",
        "drain_latch_refuses_new_work",
        "drain_queued_completes_and_fires",
        "drain_shutdown_cancels_queued_once",
        "envelope_404_harness",
        "envelope_409_cancel_terminal",
        "envelope_422_harness_validation",
        "envelope_422_v1_validation",
        "envelope_503_draining",
        "fault_conn_refused_retried_loud",
        "fault_dns_unresolvable_loud",
        "fault_multiaddr_delivers",
        "fault_private_literal_zero_attempts",
        "fault_private_resolved_refused",
        "fault_read_timeout_retried",
        "fault_tls_mismatch_retried",
        "fire_abatch_ended_once",
        "fire_abatch_gets_no_refire",
        "fire_batch_completed_once",
        "fire_batch_gets_no_refire",
        "fire_eval_terminal_once",
        "fire_ft_success_once",
        "fire_job_cancel_terminal_409",
        "fire_job_gets_no_refire",
        "fire_job_queued_cancel_once",
        "fire_job_repeated_cancel_no_refire",
        "fire_job_success_once",
        "ledger_idem_survives_restart",
        "ledger_recovered_queued_fails_closed",
        "ledger_restart_no_refire",
        "ledger_restart_restores_verdict",
        "ledger_secret_never_journaled",
        "ledger_terminal_verdict_journaled",
        "ordering_cancel_parallel_request_threads",
        "ordering_parallel_dispatch",
        "ordering_serial_fifo",
        "record_fields_batch",
        "record_fields_failed_verdict",
        "record_fields_ft",
        "record_fields_honest_job",
        "record_fields_no_callback_honest",
        "sig_delivered_both_secrets",
        "sig_format_sha256_hex",
        "sig_header_names_exact",
        "sig_rotation_secret_isolated",
        "sig_target_preserves_query",
        "sig_timestamp_unix_fresh",
        "sig_unsigned_headers_absent",
        "sig_verifies_each_hit",
        "timing_cancel_blocks_until_delivered",
        "timing_failed_job_fires_signed_failed",
        "timing_slot_held_submit_refused",
        "timing_slot_held_through_retries",
        "timing_submit_returns_before_delivery",
        "timing_terminal_before_verdict",
        "validate_accepts_http_https",
        "validate_deliver_never_raises",
        "validate_loopback_gate_env",
        "validate_refuses_bad_urls",
        "validate_secret_requires_url_422",
        "verdict_2xx_delivered_once",
        "verdict_3xx_retried_never_followed",
        "verdict_429_definitive_despite_retry_after",
        "verdict_4xx_definitive_once",
        "verdict_5xx_retried_bounded",
        "verdict_error_names_status",
    }
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = webhookdel_audit()
    assert set(results) == _EXPECTED_PROBES
    assert len(results) == 75
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = webhookdel_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = webhookdel_audit_bench()
    b = webhookdel_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def _blob_oid(rev: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", f"{rev}:{_MODULE}"],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def test_committed_receipt_binds_head_source() -> None:
    """The sealed receipt must name a revision whose audit source is the
    exact file at HEAD — a receipt sealed before a refactor is stale."""
    blob = json.loads(_RECEIPT.read_text())
    rev = blob["git_revision"]
    assert _blob_oid(rev), f"receipt names unresolvable revision {rev}"
    assert _blob_oid(rev) == _blob_oid("HEAD")
