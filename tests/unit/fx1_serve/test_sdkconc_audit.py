"""Tests for fx1.serve.sdkconc_audit — the SDK-concurrency battery."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import sdkconc_audit as audit
from fx1.serve.sdkconc_audit import sdkconc_audit, sdkconc_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the expensive end-to-end battery once per module."""
    return sdkconc_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) == len(audit._EXPECTED_PROBES)
    assert set(results) == audit._EXPECTED_PROBES
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_writer_enforcement_seams(results: dict[str, Any]) -> None:
    for name in (
        "identity_second_live_bind_refused",
        "journal_second_writer_refused",
        "journal_replay_after_close",
        "journal_post_release_mints",
        "journal_gc_releases_claim",
    ):
        assert results[name] is True, f"single-writer seam {name} uncovered"


def test_parallel_helper_fails_closed_on_unfinished_lane() -> None:
    """A wedged worker must surface as an error — never a silent
    (None, None) that ``all(e is None)`` assertions wave through."""
    import unittest.mock as mock

    orig_join = threading.Thread.join

    def quick_join(self: threading.Thread, timeout: float | None = None) -> None:
        orig_join(self, 0.05)  # shrink the join budget, keep the check

    gate = threading.Event()

    def wedge(i: int) -> str:
        if i == 0:
            gate.wait(10)  # parks well past the (patched) join budget
        return f"lane{i}"

    with mock.patch.object(threading.Thread, "join", quick_join):
        lanes = audit._parallel(wedge, n=2)
    try:
        unfinished = [e for _, e in lanes if isinstance(e, TimeoutError)]
        assert len(unfinished) == 1
        assert lanes[1] == ("lane1", None)
    finally:
        gate.set()  # release the daemon lane


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = sdkconc_audit_bench(results)
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    first = sdkconc_audit_bench(results)["receipt_sha256"]
    second = sdkconc_audit_bench(results)["receipt_sha256"]
    assert first == second


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
    blob = sdkconc_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_receipt_fails_closed_on_partial_battery() -> None:
    """An all-True subset is not a pass: the seal requires the exact
    pinned key set, so a dropped probe can never read as success."""
    partial = dict.fromkeys(sorted(audit._EXPECTED_PROBES)[:-1], True)
    assert len(partial) == len(audit._EXPECTED_PROBES) - 1
    assert sdkconc_audit_bench(partial)["claim"]["ok"] is False
    oversized = {**dict.fromkeys(audit._EXPECTED_PROBES, True), "bonus_probe": True}
    assert sdkconc_audit_bench(oversized)["claim"]["ok"] is False


def test_committed_receipt_still_verifies() -> None:
    path = Path("receipts/fx1_sdkconc_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_committed_receipt_matches_fresh_run(results: dict[str, Any]) -> None:
    """Committed-vs-fresh provenance: the sealed claim must equal a live
    battery on the same tree — semantic equivalence, not hash equality."""
    payload = json.loads(Path("receipts/fx1_sdkconc_audit.json").read_text())
    assert payload["claim"]["results"] == results
    assert payload["claim"]["ok"] is True


def test_committed_receipt_binds_head_source() -> None:
    """Provenance: the receipt's ``git_revision`` must name a commit whose
    ``sdkconc_audit.py`` blob equals this checkout's — a stale receipt
    bound to a pre-source base fails the suite. Skipped when the clone
    can't resolve the revision (exports, shallow trees)."""
    payload = json.loads(Path("receipts/fx1_sdkconc_audit.json").read_text())
    rev = payload.get("git_revision")
    assert isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev)
    show = subprocess.run(
        ["git", "show", f"{rev}:src/fx1/serve/sdkconc_audit.py"],
        capture_output=True,
        check=False,
    )
    if show.returncode != 0:
        pytest.skip("receipt revision not resolvable in this clone")
    head_source = Path("src/fx1/serve/sdkconc_audit.py").read_bytes()
    assert show.stdout == head_source, (
        f"receipt bound to {rev[:12]} whose audit source differs from HEAD"
    )


def test_battery_immune_to_ambient_state_dir(
    results: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller that exported ``FX1_SDK_STATE_DIR`` must not bind every
    state_dir=None harness to one dir (the single-writer refusal would
    then fail the battery): the audit context normalizes it and
    restores the caller's value verbatim."""
    import os

    monkeypatch.setenv("FX1_SDK_STATE_DIR", str(tmp_path))
    ambient_run = sdkconc_audit()
    assert ambient_run == results
    assert os.environ["FX1_SDK_STATE_DIR"] == str(tmp_path)
