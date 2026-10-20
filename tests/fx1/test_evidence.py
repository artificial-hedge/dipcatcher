"""Evidence-class gate: verification claims need a verified receipt path."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.honesty import validate_fx1_output
from fx1.interactive.evidence import EvidenceError, ground_summary

_PHRASES = (
    "verified receipt",
    "receipt verified",
    "independently verified",
    "verification passed",
)


def _folded(text: str) -> str:
    return " ".join(text.lower().split())


def _assert_claims_gone(text: str) -> None:
    folded = _folded(text)
    for phrase in _PHRASES:
        assert phrase not in folded
    assert "verified=true" not in folded.replace(" ", "")
    assert "the result is unverified and partial" in folded


def test_empty_summary_returns_empty_string() -> None:
    assert ground_summary("") == ""
    assert ground_summary("", secrets=["abcdef"], verified_receipts={"receipts/a.json"}) == ""


def test_verified_claim_without_verified_receipts_is_rewritten() -> None:
    summary = "The run has a verified receipt for CRPS 0.42."
    assert validate_fx1_output(summary) == summary
    out = ground_summary(summary)
    assert "verified receipt" not in _folded(out)
    assert "a the result" not in _folded(out)
    _assert_claims_gone(out)
    assert "CRPS 0.42" in out
    assert validate_fx1_output(out) == out


@pytest.mark.parametrize(
    "summary",
    [
        "The receipt verified the lane.",
        "This was independently verified.",
        "Verification passed on the lane.",
        "status verified=true",
        "status verified = TRUE",
        "VERIFIED RECEIPT of the lane.",
    ],
)
def test_verification_phrases_are_rewritten(summary: str) -> None:
    out = ground_summary(summary)
    _assert_claims_gone(out)
    assert validate_fx1_output(out) == out


def test_verified_claim_naming_verified_path_is_kept() -> None:
    path = "receipts/ok.json"
    summary = f"The verified receipt {path} records CRPS 0.42."
    out = ground_summary(summary, verified_receipts={path})
    assert out == summary
    assert "verified receipt" in _folded(out)
    assert validate_fx1_output(out) == out


def test_verified_path_may_say_verify_research() -> None:
    path = "receipts/ok.json"
    summary = f"{path} passed verify-research. pinball 0.11."
    assert ground_summary(summary, verified_receipts={path}) == summary


def test_verified_equals_true_with_verified_path_is_kept() -> None:
    path = "receipts/ok.json"
    summary = f"verified=true {path}"
    assert ground_summary(summary, verified_receipts={path}) == summary


def test_wrong_receipt_does_not_ground_or_invent_a_path() -> None:
    cited = "receipts/other.json"
    summary = f"verified receipt {cited}"
    out = ground_summary(summary, verified_receipts={"receipts/ok.json"})
    _assert_claims_gone(out)
    assert "receipts/ok.json" not in out
    assert f"evidence: cited path not in the run record: {cited}" in out


def test_artifact_path_does_not_ground_verification() -> None:
    path = "runs/score.json"
    summary = f"verified receipt {path}"
    out = ground_summary(summary, artifact_paths={path})
    _assert_claims_gone(out)
    assert path in out
    assert "evidence:" not in out


def test_only_ungrounded_lines_are_rewritten() -> None:
    path = "receipts/ok.json"
    summary = f"verified receipt {path}\nThis was independently verified.\n"
    out = ground_summary(summary, verified_receipts={path})
    lines = out.splitlines()
    assert lines[0] == f"verified receipt {path}"
    assert "independently verified" not in lines[1].lower()
    assert "unverified" in lines[1].lower()
    assert "partial" in lines[1].lower()


def test_unknown_json_path_adds_evidence_line() -> None:
    summary = "pinball 0.18 saved to runs/score.json"
    out = ground_summary(summary)
    assert out == (
        "pinball 0.18 saved to runs/score.json\n"
        "evidence: cited path not in the run record: runs/score.json"
    )
    assert validate_fx1_output(out) == out


def test_unknown_receipts_path_adds_evidence_line() -> None:
    summary = "see receipts/lane-1/manifest"
    out = ground_summary(summary)
    assert out == (
        "see receipts/lane-1/manifest\n"
        "evidence: cited path not in the run record: receipts/lane-1/manifest"
    )


def test_wrapped_json_path_is_cited_without_punctuation() -> None:
    summary = "wrote `runs/a.json`."
    out = ground_summary(summary)
    assert "evidence: cited path not in the run record: runs/a.json" in out
    assert out.count("evidence: cited path not in the run record:") == 1


def test_known_paths_are_not_flagged() -> None:
    receipt = "receipts/ok.json"
    artifact = "runs/score.json"
    summary = f"scores in {artifact} and {receipt}"
    out = ground_summary(
        summary,
        verified_receipts={receipt},
        artifact_paths={artifact},
    )
    assert out == summary


def test_unknown_paths_keep_first_seen_order() -> None:
    summary = "a runs/a.json b receipts/b"
    out = ground_summary(summary)
    lines = out.splitlines()
    assert lines[-2] == "evidence: cited path not in the run record: runs/a.json"
    assert lines[-1] == "evidence: cited path not in the run record: receipts/b"


def test_secret_is_redacted() -> None:
    secret = "sk-live-abc123"
    summary = f"token {secret} leaked"
    assert validate_fx1_output(summary) == summary
    out = ground_summary(summary, secrets=[secret, "short"])
    assert secret not in out
    assert out == "token [redacted] leaked"
    assert validate_fx1_output(out) == out


def test_secret_shorter_than_six_is_kept() -> None:
    assert ground_summary("code abcde stays", secrets=["abcde"]) == "code abcde stays"


def test_secret_replaced_on_every_occurrence() -> None:
    secret = "supersecret"
    assert ground_summary(f"{secret} and {secret}", secrets=[secret]) == (
        "[redacted] and [redacted]"
    )


def test_longer_secret_is_preferred_when_overlapping() -> None:
    text = "prefix abcdefghij suffix"
    out = ground_summary(text, secrets=["abcdef", "abcdefghij"])
    assert out == "prefix [redacted] suffix"


def test_secret_and_unverified_claim() -> None:
    secret = "supersecretvalue"
    out = ground_summary(f"verified receipt uses token {secret}", secrets=[secret])
    assert secret not in out
    assert "[redacted]" in out
    _assert_claims_gone(out)


@pytest.mark.parametrize(
    "summary",
    [
        "Sharpe",
        "the sortino discussion",
        "Calmar stays off the card",
        "headline P&L",
        "p & l",
        "NAV",
        "pnl",
        "PnL is not a score",
    ],
)
def test_forbidden_headline_tokens_raise(summary: str) -> None:
    with pytest.raises(EvidenceError):
        ground_summary(summary)


def test_sharpe_in_summary_raises_evidence_error() -> None:
    with pytest.raises(EvidenceError, match="sharpe"):
        ground_summary("The Sharpe of the sleeve is not a proper score.")


def test_sharpe_raises_even_when_a_receipt_was_verified() -> None:
    path = "receipts/ok.json"
    with pytest.raises(EvidenceError):
        ground_summary(
            f"verified receipt {path} and Sharpe 1.2",
            verified_receipts={path},
        )


def test_evidence_error_is_a_value_error() -> None:
    assert issubclass(EvidenceError, ValueError)


@pytest.mark.parametrize(
    "summary",
    [
        "navigation of the canvas",
        "sharpened pinball",
        "partial calibration",
        "the panel scored CRPS 0.2",
    ],
)
def test_ordinary_words_are_not_forbidden_tokens(summary: str) -> None:
    assert ground_summary(summary) == summary
    assert validate_fx1_output(summary) == summary


def test_partial_unverified_text_passes_through() -> None:
    summary = (
        "The result is partial and unverified. CRPS 0.41 and pinball 0.2; "
        "PIT was checked on the lane."
    )
    assert ground_summary(summary) == summary
    assert validate_fx1_output(summary) == summary


def test_blank_lines_and_crlf_pass_through() -> None:
    summary = "partial\r\n\r\nCRPS 0.2\r\n"
    assert ground_summary(summary) == summary


def test_synthetic_label_is_preserved() -> None:
    summary = "On SYNTHETIC data the model achieves 0.99 recovery accuracy."
    assert ground_summary(summary) == summary
    assert "SYNTHETIC" in summary
    assert validate_fx1_output(summary) == summary


def test_synthetic_label_survives_claim_rewrite() -> None:
    summary = (
        "On SYNTHETIC data a verified receipt shows recovery accuracy 0.99 as a correctness check."
    )
    assert validate_fx1_output(summary) == summary
    out = ground_summary(summary)
    assert "SYNTHETIC" in out
    assert "synthetic" not in out.replace("SYNTHETIC", "")
    _assert_claims_gone(out)
    assert validate_fx1_output(out) == out


def test_does_not_invent_paths() -> None:
    out = ground_summary("The result is partial. CRPS 0.2.")
    assert out == "The result is partial. CRPS 0.2."
    assert ".json" not in out
    assert "receipts/" not in out
    assert "verify-research" not in out


def test_module_does_not_import_quant_fund_or_touch_io() -> None:
    import fx1.interactive.evidence as evidence

    source = Path(evidence.__file__).read_text(encoding="utf-8")
    assert "quant_fund" not in source
    assert "is_file" not in source
    assert "urlopen" not in source
    assert "requests" not in source
