"""Tests for fx1.serve.webhook_delivery_audit — dispatcher internals."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import webhook_delivery_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the scripted-server battery once per module — loopback
    http.server + patched clocks, deterministic."""
    return audit.webhook_delivery_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) == len(audit._EXPECTED_PROBES)
    assert set(results) == audit._EXPECTED_PROBES
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_dispatcher_seams(results: dict[str, Any]) -> None:
    for name in (
        "verify_rejects_reserialized_body",
        "signed_stale_still_refused",
        "literal_private_refused_0",
        "env_allows_loopback_literal",
        "env_restored_after_leg",
        "v4_mapped_private_refused",
        "mixed_public_private_refused",
        "resolved_cgnat_refused",
        "resolved_ula_refused",
        "resolved_17216_refused",
        "http_connect_dials_validated_ip",
        "https_sni_is_url_host",
        "fail_fail_ok_delivers_third",
        "backoff_doubles",
        "definitive_4xx_no_retry",
        "fresh_ts_per_attempt",
        "fault_walks_to_next_address",
        "five_xx_short_circuits_walk",
        "conn_refused_retries_then_errors",
        "private_url_refused_zero_attempts",
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
    blob = audit.webhook_delivery_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_receipt_fails_closed_on_partial_battery() -> None:
    """An all-True subset is not a pass: the seal requires the exact
    pinned key set, so a dropped probe can never read as success."""
    partial = dict.fromkeys(sorted(audit._EXPECTED_PROBES)[:-1], True)
    assert len(partial) == len(audit._EXPECTED_PROBES) - 1
    assert audit.webhook_delivery_audit_bench(partial)["claim"]["ok"] is False
    oversized = {**dict.fromkeys(audit._EXPECTED_PROBES, True), "bonus_probe": True}
    assert audit.webhook_delivery_audit_bench(oversized)["claim"]["ok"] is False


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = audit.webhook_delivery_audit_bench(results)
    assert blob["claim"]["ok"] is True
    assert verify_receipt_payload(blob)["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = audit.webhook_delivery_audit_bench(results)
    b = audit.webhook_delivery_audit_bench(results)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_committed_receipt_verifies_without_rerunning() -> None:
    path = Path("receipts/fx1_webhook_delivery_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_committed_receipt_matches_fresh_run(results: dict[str, Any]) -> None:
    """Committed-vs-fresh provenance: the sealed claim must equal a live
    battery on the same tree — semantic equivalence, not hash equality."""
    payload = json.loads(Path("receipts/fx1_webhook_delivery_audit.json").read_text())
    assert payload["claim"]["results"] == results
    assert payload["claim"]["ok"] is True


def test_committed_receipt_binds_head_source() -> None:
    """Provenance: the receipt's ``git_revision`` must name a commit whose
    ``webhook_delivery_audit.py`` blob equals this checkout's — a stale
    receipt bound to a pre-source base fails the suite. Skipped when the
    clone can't resolve the revision (exports, shallow trees)."""
    payload = json.loads(Path("receipts/fx1_webhook_delivery_audit.json").read_text())
    rev = payload.get("git_revision")
    assert isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev)
    show = subprocess.run(
        ["git", "show", f"{rev}:src/fx1/serve/webhook_delivery_audit.py"],
        capture_output=True,
        check=False,
    )
    if show.returncode != 0:
        pytest.skip("receipt revision not resolvable in this clone")
    head_source = Path("src/fx1/serve/webhook_delivery_audit.py").read_bytes()
    assert show.stdout == head_source, (
        f"receipt bound to {rev[:12]} whose audit source differs from HEAD"
    )


def test_private_env_never_leaks() -> None:
    """The SSRF opt-in must not survive the battery — the caller's env
    (present or absent) is restored verbatim, never mutated."""
    import os

    sentinel = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    audit.webhook_delivery_audit()
    assert os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS") == sentinel


@pytest.mark.parametrize("ambient", ["1", "yes", "0"])
def test_battery_immune_to_ambient_opt_in(
    results: dict[str, Any], monkeypatch: pytest.MonkeyPatch, ambient: str
) -> None:
    """A caller that legitimately exported the SSRF opt-in must not
    change the measured contract: the battery normalizes env per leg
    and restores the caller's value verbatim afterwards."""
    import os

    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", ambient)
    ambient_run = audit.webhook_delivery_audit()
    assert ambient_run == results
    assert os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] == ambient


def test_battery_restores_absent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the opt-in absent, the battery leaves it absent."""
    import os

    monkeypatch.delenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", raising=False)
    audit.webhook_delivery_audit()
    assert "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS" not in os.environ
