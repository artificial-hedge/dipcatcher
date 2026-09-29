"""Honesty-inheritance tests: fx-1's output contract, tested like code."""

import time

import pytest

from fx1.honesty import Fx1HonestyError, validate_fx1_output


def test_clean_research_output_passes():
    text = (
        "The distribution family scored CRPS 0.42 and PIT uniformity passed; "
        "verify with `uv run dipcatcher verify-research`."
    )
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize("token", ["sharpe", "sortino", "calmar", "pnl", "nav"])
def test_forbidden_headline_metric_fails_closed(token: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(f"The strategy achieved {token}: 2.35 over the panel.")


def test_bare_discussion_of_forbidden_metrics_allowed():
    # Explaining *why* a metric is forbidden is not a headline claim.
    text = "The lab forbids sharpe headlines in research scorecards by construction."
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize(
    "claim",
    [
        "This produced live P&L of $12,000 last quarter.",
        "These are real money returns you can expect.",
        "The strategy offers guaranteed returns.",
    ],
)
def test_live_performance_claims_fail_closed(claim: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(claim)


def test_synthetic_evidence_requires_label():
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output("On synthetic data the model achieves 0.99 recovery accuracy.")
    # Explicit SYNTHETIC label passes.
    assert validate_fx1_output("On SYNTHETIC data the model achieves 0.99 recovery accuracy.")


def test_synthetic_commentary_without_numbers_allowed():
    # Refusals and contract discussion mention synthetic evidence without
    # presenting results - not a violation.
    text = "I will not rename synthetic evidence; the label stays."
    assert validate_fx1_output(text) == text


# ---------------------------------------------------------------------------
# (a) word-boundary correctness: whole-word tokens only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "We navigate the config panel cleanly; nothing numeric is headlined.",
        "navel orange weighing 1.5 kg sits on the counter",
        "sharpen the pencil to 2.1 mm",
        "The navel rating was 2.4 - a completely different word.",
        "We navigate to 2.1 km away.",
    ],
)
def test_substring_lookalikes_do_not_fire(text: str):
    # A forbidden token appearing only inside a longer word is not a claim.
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize(
    "text",
    [
        # Compound receipt keys: parity bookkeeping, never headline claims.
        "nav_final_dipcatcher 1500764.65 and nav_final_qlib 1498592.66 match.",
        "nav_max_abs_diff 0.1783 and nav_max_rel_diff 1.03e-07.",
        '{"init_nav": 1000000.0, "fills_dipcatcher": 1587}',
        "Correctness is NAV parity; latency is single-process wall time.",
        "pnl_series and pnl_total are lab-side column names, not results.",
    ],
)
def test_compound_underscore_keys_do_not_fire(text: str):
    # Underscore counts as a word character, mirroring `\b` semantics and the
    # catalog's underscore-token key rule.
    assert validate_fx1_output(text) == text


def test_token_followed_by_letters_does_not_fire():
    # "nav2" is a compound key, not a headline claim.
    text = "nav2 is a column name in the receipt table."
    assert validate_fx1_output(text) == text


# ---------------------------------------------------------------------------
# (b) false negatives: headline phrasings that must be caught
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Sharpe ratio: 2.35",
        "sharpe-ratio: 2.35",
        "Sharpe ratio came in at 2.35 over the panel.",
        "The Sharpe is 2.4 - headline result.",
        "sharpe ratio of 2.35",
        "(Sharpe) = 2.4;",
        '"sharpe": 2.4',
        "--SHARPE-- 2.4",
        "Sharpe\n\n2.4",
        "sharpe: $2.40",
        "sharpe 61%",
        "sortino ratio reached 3.1",
        "calmar value was 2.2",
        "P&L was $4,200 last month.",
        "profit and loss peaked at 4200 dollars.",
        "net asset value peaked at 1.9",
        "The model's P&L: +12,000.",
        "2.35 was the Sharpe we achieved.",
        "1.9 is its nav multiple.",
        "0.99 equals the sortino reading.",
    ],
)
def test_headline_metric_phrasings_fail_closed(text: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize(
    "text",
    [
        "We tested 3 sharpe variants and 5 sortino ones.",
        "top 5 nav strategies were dropped for leakage",
        "sharpe, 3 others were considered",
        "P&L; research results are proper scores.",
        "the nav. 2 of 5 panels was rerun",
        "3 of 5 receipts report crps 0.42",
    ],
)
def test_counting_and_prose_do_not_fire(text: str):
    # Numbers that merely sit near a token (counts, enumerations, prose) are
    # mentions, not headline claims.
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "I cannot headline a sharpe figure - the honesty contract forbids "
        "forbidden headline metrics; research results are proper scores "
        "(CRPS, pinball, PIT).",
        "I cannot report a P&L figure - headline P&L is forbidden; research "
        "results are proper scores.",
        "I cannot headline a NAV figure - headline NAV claims are forbidden.",
        "The lab reports proper scores, not Sharpe headlines; this family's "
        "CRPS and PIT results are in the receipt.",
    ],
)
def test_refusals_stay_clean(text: str):
    # The mention-vs-claim contract: a refusal that explains the rule is not
    # itself a violation, even though it names the token.
    assert validate_fx1_output(text) == text


# ---------------------------------------------------------------------------
# (c) unicode evasion: homoglyphs, full-width, zero-width / format chars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Cyrillic homoglyphs (NFKC does NOT fold these; the confusable map does)
        "Sh\u0430rpe ratio is 2.1",  # Cyrillic а
        "\u0405harpe is 2.1",  # Cyrillic Ѕ (uppercase)
        "\u0455harpe is 2.1",  # Cyrillic ѕ (lowercase)
        "Sh\u0430r\u0440e 2.1",  # Cyrillic а + р
        "\u0441almar is 2.2",  # Cyrillic с for c
        "\u0455ortino 3.1",  # Cyrillic ѕ for s
        "n\u0430v 1.5",  # Cyrillic а inside latin "nav"
        # Greek homoglyphs
        "\u03bdav 1.5",  # Greek ν for n
        "\u03c1nl 4.2",  # Greek ρ for p
        "sh\u03b1rpe 2.1",  # Greek α for a
        # Full-width ASCII (NFKC folds these)
        "\uff33\uff48\uff41\uff52\uff50\uff45 is 2.1",  # Ｓｈａｒｐｅ
        "\uff4e\uff41\uff56 1.5",  # ｎａｖ
        "\uff50\uff4e\uff4c 4.2",  # ｐｎｌ
        # Zero-width / format characters (Cf category, survive NFKC)
        "sha\u200brpe is 2.1",  # ZWSP
        "shar\u200cpe is 2.1",  # ZWNJ
        "nav\u200d 1.5",  # ZWJ
        "s\u2060harpe 2.1",  # word joiner
        "\ufeffsharpe 2.1",  # BOM
        "shar\u00adpe 2.1",  # soft hyphen
        # Full-width digits and punctuation
        "\uff33\uff48\uff41\uff52\uff50\uff45\uff1a\u00a0\uff12\uff0e\uff14",
    ],
)
def test_unicode_evasions_fail_closed(text: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


def test_normalization_never_alters_returned_text():
    # The normalized copy is for matching only; clean text is returned verbatim.
    text = "CRPS 0.42 and PIT uniformity passed."
    assert validate_fx1_output(text) == text


def test_normalization_is_idempotent():
    from fx1.honesty import _normalize_for_match

    once = _normalize_for_match("Sh\u0430rpe\u200b ratio 2.1")
    twice = _normalize_for_match(once)
    assert once == twice == "sharpe ratio 2.1"


def test_confusable_fold_is_ascii_targeted_and_deterministic():
    from fx1.honesty import _CONFUSABLE_TABLE, _normalize_for_match

    # Every mapped target is plain ASCII, so folding cannot introduce new
    # homoglyphs or grow the alphabet.
    assert all(ord(dst) <= 127 for dst in _CONFUSABLE_TABLE.values())
    for _ in range(3):
        assert _normalize_for_match("\u0455\u04bb\u0430rpe") == "sharpe"


# ---------------------------------------------------------------------------
# (d) de-spelling and separator evasions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "S h a r p e ratio is 2.35",
        "s-h-a-r-p-e equals 2.35",
        "s_h_a_r_p_e 2.35",
        "n a v 1.5",
        "n-a-v: 1.5",
        "p n l 4.2",
        "s o r t i n o 3.1",
    ],
)
def test_despelled_tokens_fail_closed(text: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize(
    "text",
    [
        "n a v e l orange is a fruit",  # de-spelling cannot revive "navel"
        "n a v i g a t e the panel",
        "s h a r p e n the pencil to 2.1 mm",
    ],
)
def test_despelled_lookalikes_do_not_fire(text: str):
    assert validate_fx1_output(text) == text


# ---------------------------------------------------------------------------
# (e) case / whitespace / punctuation invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "SHARPE 2.4",
        "ShArPe 2.4",
        "sharpe 2.4",
        "\tsharpe\t2.4",
        "sharpe    2.4",
        "sharpe:2.4",
        "sharpe = 2.4",
        "sharpe -> 2.4",
        "sharpe) 2.4",
        "sharpe] 2.4",
        "sharpe} 2.4",
        "sharpe' 2.4",
        'sharpe" 2.4',
    ],
)
def test_case_whitespace_punctuation_invariance(text: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize("variant", ["Sharpe", "SHARPE", "sharpe", "ShArPe", "sHaRpE"])
def test_case_insensitive_across_all_variants(variant: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(f"{variant} 2.4 is the headline")


# ---------------------------------------------------------------------------
# (f) matcher equivalence / determinism / safety
# ---------------------------------------------------------------------------


def test_matcher_is_deterministic():
    text = "Sharpe ratio came in at 2.35 over the panel."
    outcomes = []
    for _ in range(5):
        try:
            validate_fx1_output(text)
            outcomes.append("pass")
        except Fx1HonestyError:
            outcomes.append("block")
    assert outcomes == ["block"] * 5


def test_matcher_is_linear_time_on_adversarial_input():
    # The connector run is bounded and separator runs are bounded, so a long
    # run of near-miss separators cannot blow up into catastrophic
    # backtracking. This guards the ReDoS surface of the hardened matcher.
    adversarial = "sharpe" + " -" * 5000 + " 2.4"
    started = time.perf_counter()
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(adversarial)
    assert time.perf_counter() - started < 2.0

    pathological = "s h a r p e " * 2000 + "no number here"
    started = time.perf_counter()
    assert validate_fx1_output(pathological) == pathological
    assert time.perf_counter() - started < 2.0


def test_every_pattern_compiles_and_alias_table_covers_every_token():
    from fx1.honesty import _HEADLINE_PATTERNS, FORBIDDEN_HEADLINE_TOKENS

    # Patterns are precompiled at import; assert the table is non-empty and
    # covers every mirrored token in both directions.
    covered = {token for token, _ in _HEADLINE_PATTERNS}
    assert covered == set(FORBIDDEN_HEADLINE_TOKENS)
    # Two builders (forward + inverse) per alias spelling.
    assert len(_HEADLINE_PATTERNS) >= 2 * len(FORBIDDEN_HEADLINE_TOKENS)


def test_error_message_names_the_offending_token():
    with pytest.raises(Fx1HonestyError) as excinfo:
        validate_fx1_output("Sharpe ratio: 2.35")
    assert "'sharpe'" in str(excinfo.value)
