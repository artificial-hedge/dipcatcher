"""Trigger lexicon: fast planning matches legacy word-boundary semantics."""

from __future__ import annotations

import re

from fx1.interactive.superpower import _TRIGGERS, _WEB_TRIGGERS, _score_goal, plan_goal
from fx1.interactive.trigger_lexicon import GoalFold, TriggerLexicon


def _legacy_score(goal: str, terms: tuple[str, ...]) -> int:
    lowered = goal.lower()
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))


def test_lexicon_scores_match_legacy_for_sample_goals() -> None:
    lex = TriggerLexicon(_TRIGGERS)
    goals = (
        "run the conformal volatility benchmark family",
        "check the northset order book identities",
        "what is the latest news on the Fed rate decision",
        "ingest data then build features",
    )
    for goal in goals:
        lowered = goal.lower()
        for _cap_id, terms in _TRIGGERS.items():
            assert lex.score(lowered, terms) == _legacy_score(goal, terms)
            assert _score_goal(goal, terms) == _legacy_score(goal, terms)


def test_web_trigger_parity() -> None:
    lex = TriggerLexicon(_TRIGGERS)
    goal = "look up recent literature on conformal prediction"
    assert lex.any_web_trigger(goal.lower(), _WEB_TRIGGERS)
    assert _score_goal(goal, _WEB_TRIGGERS) > 0


def test_goal_fold_reuses_token_index_for_ranking() -> None:
    lex = TriggerLexicon(_TRIGGERS)
    fold = GoalFold.fold("verify receipts and run doctor health check")
    harness_ids = tuple(_TRIGGERS)
    ranked = lex.rank_harness_folded(fold, harness_ids, max_harness=4)
    assert ranked
    assert lex.rank_harness(fold.lowered, harness_ids, max_harness=4) == ranked


def test_plan_still_selects_research_and_web() -> None:
    bench = plan_goal("run the conformal volatility benchmark family")
    assert "research" in bench.ids()
    web = plan_goal("what is the latest news on the Fed rate decision")
    assert "web_research" in web.ids()
