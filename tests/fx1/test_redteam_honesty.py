"""Honesty gate, unverified prose, and forged receipts."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.interactive.evidence import EvidenceError, ground_summary
from fx1.interactive.evidence_ledger import Verdict, verify_receipt
from tests.fx1.redteam_support import completion, isolate, joined, open_chat


@pytest.fixture()
def root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate(monkeypatch, tmp_path)
    return tmp_path


def _answer(root: Path, text: str) -> tuple[str, str]:
    concierge, out, _harness = open_chat(root, [completion(text, None)])
    concierge.handle_line("report")
    return concierge.last_status, joined(out)


def test_sharpe_pnl_nav_and_guaranteed_profits_are_withheld(root: Path) -> None:
    samples = (
        "Sharpe 3.2",
        "sharpe\u20603.2",
        "The P&L is 4.2",
        "NAV 1.9",
        "guaranteed profits of 2",
    )
    for text in samples:
        status, shown = _answer(root, text)
        assert status == "withheld", text
        assert "3.2" not in shown
        assert "4.2" not in shown
        assert "withheld" in shown


def test_unlabeled_synthetic_is_withheld_and_labeled_synthetic_is_kept(root: Path) -> None:
    status, shown = _answer(root, "The synthetic backtest shows a score of 0.91")
    assert status == "withheld"
    assert "0.91" not in shown
    status, shown = _answer(root, "SYNTHETIC QLIKE is 0.42")
    assert status == "ok"
    assert "0.42" in shown
    assert "SYNTHETIC" in shown


def test_ground_summary_downgrades_unverified_prose_and_unknown_paths() -> None:
    grounded = ground_summary("This result is verified.", verified_receipts=set())
    assert "unverified" in grounded.casefold()
    assert "verified." not in grounded.casefold()
    cited = ground_summary(
        "verified receipt receipts/forged.json",
        verified_receipts=set(),
    )
    assert "not in the run record" in cited
    with pytest.raises(EvidenceError):
        ground_summary("Sharpe 3.2", verified_receipts=set())


def test_forged_receipt_paths_are_not_verified(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    fake = run / "not_a_receipt.json"
    fake.write_text('{"verified": true, "dataset_sha256": "' + ("ab" * 32) + '"}', encoding="utf-8")
    item = verify_receipt(fake, roots=[run])
    assert item.verdict is not Verdict.VERIFIED

    outside = tmp_path / "outside" / "real.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    link = run / "linked.json"
    link.symlink_to(outside)
    linked = verify_receipt(link, roots=[run])
    assert linked.verdict is not Verdict.VERIFIED

    escaped = verify_receipt(outside, roots=[run])
    assert escaped.verdict is not Verdict.VERIFIED
