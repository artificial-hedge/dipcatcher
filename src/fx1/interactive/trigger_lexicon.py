"""Compiled trigger lexicon for /superpower planning.

Goals are scored against hundreds of multi-word triggers across every harness
capability. Recompiling word-boundary regexes on every plan would dominate
interactive latency; this module builds the patterns once and reuses them for
the lifetime of the process.

Matching semantics are identical to the legacy ``\\b{term}\\b`` scan over a
lower-cased goal: single-token triggers hit the unigram index; phrases keep
their own compiled patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache


@lru_cache(maxsize=2048)
def _word_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(term.lower())}\b")


def _tokenize(lowered: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", lowered))


@dataclass(frozen=True, slots=True)
class GoalFold:
    """One lower-cased goal with token index materialised once per planning pass."""

    raw: str
    lowered: str
    tokens: frozenset[str]

    @classmethod
    def fold(cls, goal: str) -> GoalFold:
        lowered = goal.lower()
        return cls(raw=goal, lowered=lowered, tokens=_tokenize(lowered))

    def score_terms(self, terms: tuple[str, ...]) -> int:
        if not terms:
            return 0
        hits = 0
        for term in terms:
            parts = term.lower().split()
            if len(parts) == 1:
                if parts[0] in self.tokens:
                    hits += 1
            elif _word_pattern(term).search(self.lowered):
                hits += 1
        return hits

    def matched_terms(self, terms: tuple[str, ...]) -> list[str]:
        if not terms:
            return []
        out: list[str] = []
        for term in terms:
            parts = term.lower().split()
            if len(parts) == 1:
                if parts[0] in self.tokens:
                    out.append(term)
            elif _word_pattern(term).search(self.lowered):
                out.append(term)
        return out


@dataclass(frozen=True)
class TriggerLexicon:
    """Immutable, pre-indexed trigger map keyed by capability id."""

    triggers: dict[str, tuple[str, ...]]
    _unigram: dict[str, tuple[tuple[str, str], ...]] = field(init=False, repr=False)
    _phrases: tuple[tuple[re.Pattern[str], str, str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        unigram: dict[str, list[tuple[str, str]]] = {}
        phrases: list[tuple[re.Pattern[str], str, str]] = []
        for cap_id, terms in self.triggers.items():
            for term in terms:
                parts = term.lower().split()
                if len(parts) == 1:
                    unigram.setdefault(parts[0], []).append((cap_id, term))
                else:
                    phrases.append((_word_pattern(term), cap_id, term))
        object.__setattr__(
            self,
            "_unigram",
            {token: tuple(entries) for token, entries in unigram.items()},
        )
        object.__setattr__(self, "_phrases", tuple(phrases))

    def score(self, lowered_goal: str, terms: tuple[str, ...]) -> int:
        """Count distinct trigger terms from *terms* that appear in *lowered_goal*."""
        if not terms:
            return 0
        tokens = _tokenize(lowered_goal)
        hits = 0
        for term in terms:
            parts = term.lower().split()
            if len(parts) == 1:
                if parts[0] in tokens:
                    hits += 1
            elif _word_pattern(term).search(lowered_goal):
                hits += 1
        return hits

    def matched(self, lowered_goal: str, terms: tuple[str, ...]) -> list[str]:
        """Return trigger terms that matched, preserving declaration order."""
        if not terms:
            return []
        tokens = _tokenize(lowered_goal)
        out: list[str] = []
        for term in terms:
            parts = term.lower().split()
            if len(parts) == 1:
                if parts[0] in tokens:
                    out.append(term)
            elif _word_pattern(term).search(lowered_goal):
                out.append(term)
        return out

    def fold(self, goal: str) -> GoalFold:
        return GoalFold.fold(goal)

    def rank_harness(
        self,
        lowered_goal: str,
        harness_ids: tuple[str, ...],
        *,
        max_harness: int,
    ) -> list[str]:
        """Score only *harness_ids* and return the top *max_harness* ids."""
        return self.rank_harness_folded(
            GoalFold.fold(lowered_goal), harness_ids, max_harness=max_harness
        )

    def rank_harness_folded(
        self,
        fold: GoalFold,
        harness_ids: tuple[str, ...],
        *,
        max_harness: int,
    ) -> list[str]:
        """Like :meth:`rank_harness` but reuses a pre-folded goal surface."""
        scored: list[tuple[int, str]] = []
        for cap_id in harness_ids:
            terms = self.triggers.get(cap_id, ())
            score = fold.score_terms(terms)
            if score:
                scored.append((score, cap_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [cap_id for _, cap_id in scored[:max_harness]]

    def any_web_trigger(self, lowered_goal: str, web_terms: tuple[str, ...]) -> bool:
        return self.score(lowered_goal, web_terms) > 0

    def any_web_trigger_folded(self, fold: GoalFold, web_terms: tuple[str, ...]) -> bool:
        return fold.score_terms(web_terms) > 0
