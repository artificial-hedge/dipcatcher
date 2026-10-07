"""Tests for fx1.serve.client_retry_audit — HarnessClient retry mechanics."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import client_retry_audit as audit
from fx1.serve.client import HarnessTransportError
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the scripted-transport battery once per module — pure CPU,
    no sockets or wall-clock sleeps."""
    return audit.client_retry_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) == len(audit._EXPECTED_PROBES)
    assert set(results) == audit._EXPECTED_PROBES
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_retry_schedule_seams(results: dict[str, Any]) -> None:
    for name in (
        "backoff_doubles_pure_faults",
        "fault_sleeps_capped_at_max_wait",
        "ra_overrides_grown_backoff",
        "ra_at_cap_retries",
        "ra_over_cap_breaks_no_sleep",
        "sleep_precedes_retry",
        "mapped_503_keeps_headers",
        "mapped_500_clears_headers",
        "mapped_503_exhaustion_skips_circuit",
        "idempotency_key_on_every_attempt",
    ):
        assert results[name] is True, f"seam {name} uncovered"


@pytest.mark.parametrize(
    "invalid_results",
    [
        {},
        {"probe": False},
        {"probe": 1},
        {"probe": None},
    ],
)
def test_receipt_fails_closed_on_invalid_results(invalid_results: dict[str, Any]) -> None:
    blob = audit.client_retry_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_receipt_fails_closed_on_partial_battery() -> None:
    """An all-True subset is not a pass: the seal requires the exact
    pinned key set, so a dropped probe can never read as success."""
    partial = dict.fromkeys(sorted(audit._EXPECTED_PROBES)[:-1], True)
    assert len(partial) == len(audit._EXPECTED_PROBES) - 1
    assert audit.client_retry_audit_bench(partial)["claim"]["ok"] is False
    oversized = {**dict.fromkeys(audit._EXPECTED_PROBES, True), "bonus_probe": True}
    assert audit.client_retry_audit_bench(oversized)["claim"]["ok"] is False


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = audit.client_retry_audit_bench(results)
    assert blob["claim"]["ok"] is True
    assert verify_receipt_payload(blob)["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = audit.client_retry_audit_bench(results)
    b = audit.client_retry_audit_bench(results)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_committed_receipt_verifies_without_rerunning() -> None:
    path = Path("receipts/fx1_client_retry_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_committed_receipt_matches_fresh_run(results: dict[str, Any]) -> None:
    """Committed-vs-fresh provenance: the sealed claim must equal a live
    battery on the same tree — semantic equivalence, not hash equality
    (a reseal on a different revision changes the digest legitimately)."""
    payload = json.loads(Path("receipts/fx1_client_retry_audit.json").read_text())
    assert payload["claim"]["results"] == results
    assert payload["claim"]["ok"] is True


def test_committed_receipt_revision_is_source_commit() -> None:
    """Provenance: the receipt's ``git_revision`` must name a real commit
    containing this audit's source — never a bare merge-base that
    predates it. Verified against the checkout when git can resolve
    ancestry; skipped when the clone can't (exports, shallow trees
    missing the object)."""
    payload = json.loads(Path("receipts/fx1_client_retry_audit.json").read_text())
    rev = payload.get("git_revision")
    assert isinstance(rev, str)
    assert re.fullmatch(r"[0-9a-f]{40}", rev)
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{rev}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("receipt revision not resolvable in this clone")
    source = subprocess.run(
        ["git", "ls-tree", "-r", rev, "--", "src/fx1/serve/client_retry_audit.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert source.returncode == 0
    assert "client_retry_audit.py" in source.stdout, (
        f"receipt bound to {rev[:12]} which lacks the audit source"
    )


def test_half_open_admits_single_probe_under_contention() -> None:
    """Regression: eight callers released together at a lapsed window
    must admit exactly one transport probe; the rest fail fast."""
    clock_t = [1000.0]
    from fx1.serve.client import HarnessClient

    trip, calls, _ = audit._scripted(HarnessTransportError("dial failed"))
    gate = threading.Event()
    dialed: list[tuple[str, str]] = []

    def probe_transport(method: str, url: str, payload: Any, headers: Any, timeout_s: float) -> Any:
        dialed.append((method, urllib.parse.urlparse(url).path))
        gate.wait(10.0)
        return 200, {}, audit._ITEMS_BODY

    cli = HarnessClient(
        audit._BASE,
        transport=trip,
        circuit_breaker_threshold=1,
        circuit_reset_s=30.0,
        clock=lambda: clock_t[0],
        sleep=lambda s: None,
    )
    with pytest.raises(HarnessTransportError):
        cli.commands()
    cli._transport = probe_transport  # noqa: SLF001
    clock_t[0] += 31.0

    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def racer() -> None:
        barrier.wait()
        try:
            cli.commands()
            outcomes.append("ok")
        except HarnessTransportError as exc:
            outcomes.append("circuit" if "circuit open" in str(exc) else "other")
        finally:
            if len(outcomes) == 7:
                gate.set()

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert outcomes.count("ok") == 1
    assert outcomes.count("circuit") == 7
    assert len(dialed) == 1


def test_unkeyed_write_never_retries_even_with_retry_writes() -> None:
    """Regression: an ambiguous transport fault on an unkeyed POST must
    surface once — never be replayed — even when retry_writes is set."""

    fault = HarnessTransportError("connection reset mid-write")
    tr, calls, _ = audit._scripted(fault)
    cli = audit._mk(tr, max_retries=3, retry_writes=True, sleep=lambda s: None)
    with pytest.raises(HarnessTransportError):
        cli.complete([{"role": "user", "content": "ping"}])
    assert len(calls) == 1


def test_scripted_transport_repeats_last_step() -> None:
    """The fault-forever leg: a lone exception step must repeat, not
    degrade into the 500 fallback — the retry probes lean on it."""
    fault = HarnessTransportError("dial failed")
    tr, calls, _ = audit._scripted(fault)
    cli = audit._mk(tr, max_retries=2, sleep=lambda s: None)
    exc = None
    try:
        cli.commands()
    except HarnessTransportError as e:
        exc = e
    assert exc is not None
    assert "dial failed" in str(exc)
    assert len(calls) == 3
