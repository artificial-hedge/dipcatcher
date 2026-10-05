"""Tests for fx1.serve.concurrency_audit — concurrency/idempotency battery."""

from __future__ import annotations

import pytest

from fx1.serve.concurrency_audit import concurrency_audit, concurrency_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes that pinned False as measured divergences before this lane's
# fixes. Each names the contract it now holds; the assertions below keep
# them regression-pinned True so a reversion re-breaks the test, not the
# receipt.
#
# The measured defects were:
#  - same-Idempotency-Key parallel POSTs double-executed on every
#    idempotent route whose claim span didn't cover the handler body
#    (complete, complete/batch, chat, messages, messages/batches,
#    completions, responses sync+bg, batches, ft jobs, evals) — fixed by
#    the ``_idem_guard`` Depends / ``claim_lock`` spans;
#  - a same-key replay arriving mid-execution raced a second backend
#    call instead of replaying the leader's record;
#  - ``JobJournal.append`` read seq/chain outside its lock → twin seqs
#    and a broken hash chain under parallel appends;
#  - conversation item read-modify-write (add/delete/turn-append) lost
#    updates under parallel writers — now ``mutate_items``;
#  - background-response status regressed: a queued→in_progress flip
#    clobbered a landed cancel, and a late cancel clobbered the worker's
#    terminal verdict — now CAS via ``transition_status`` /
#    ``put_unless_status``;
#  - the same-key burst double-billed the managed key twice.
_FORMER_DEFECTS = {
    "idem_race_complete",
    "idem_race_complete_batch",
    "idem_race_chat",
    "idem_race_messages",
    "idem_race_msg_batches",
    "idem_race_completions",
    "idem_race_responses",
    "idem_race_responses_bg",
    "idem_race_batches",
    "idem_race_ft_jobs",
    "idem_race_evals",
    "replay_midflight_chat",
    "replay_midflight_complete",
    "replay_midflight_responses",
    "conv_items_parallel_no_lost_append",
    "conv_item_deletes_consistent",
    "bg_cancel_midflight_terminal",
    "bg_cancel_queued_never_runs",
    "journal_parallel_append_chain_unbroken",
    "key_idem_replay_no_double_charge",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_contract_probes_hold() -> None:
    results = concurrency_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_defect_probes_fixed() -> None:
    """Each formerly pinned defect now measures the fixed contract."""
    results = concurrency_audit()
    for name in sorted(_FORMER_DEFECTS):
        assert results.get(name) is True, f"former defect {name} regressed"


def test_receipt_verifies() -> None:
    blob = concurrency_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = concurrency_audit_bench()
    b = concurrency_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
