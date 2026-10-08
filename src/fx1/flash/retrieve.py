"""Flash-context retrieval: deterministic, explainable relevance scoring.

No embedding service and no network: each entry is hashed into a sparse
vector (SHA-256 → 20-bit buckets) over its text, tags, and task words, and a
query is scored by cosine similarity, tempered by recency and staleness.
``reasons`` explains *why* an entry matched (shared terms, tag hit, task
overlap, stale flag) so retrieval is auditable — the opposite of a black box.

The scoring design deliberately prefers precision: unrelated material scores
0 against a query with no shared vocabulary and is left behind instead of
filling the model's context.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fx1.flash.store import FlashEntry, FlashStore

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "which",
        "while",
        "you",
        "your",
    }
)

VEC_DIM = 1 << 20
DEFAULT_HALF_LIFE_S = 30 * 24 * 3600  # 30 days
DEFAULT_MIN_SCORE = 0.06


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with a small stopword filter."""
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


def hash_vec(tokens: list[str], weight: float = 1.0) -> dict[int, float]:
    """Hashed sparse bag-of-words; *weight* scales every count."""
    vec: dict[int, float] = {}
    for token in tokens:
        bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % VEC_DIM
        vec[bucket] = vec.get(bucket, 0.0) + weight
    return vec


def _cosine(qvec: dict[int, float], dvec: dict[int, float]) -> float:
    num = sum(qv * dvec.get(bucket, 0.0) for bucket, qv in qvec.items())
    if not num:
        return 0.0
    qnorm = math.sqrt(sum(v * v for v in qvec.values()))
    dnorm = math.sqrt(sum(v * v for v in dvec.values()))
    if not qnorm or not dnorm:
        return 0.0
    return num / (qnorm * dnorm)


def _entry_vec(entry: FlashEntry) -> dict[int, float]:
    """Text words count once, tags twice, task words 1.5x — field-weighted."""
    vec = hash_vec(tokenize(entry.text), 1.0)
    for bucket, count in hash_vec(tokenize(" ".join(entry.tags)), 2.0).items():
        vec[bucket] = vec.get(bucket, 0.0) + count
    for bucket, count in hash_vec(tokenize(entry.task), 1.5).items():
        vec[bucket] = vec.get(bucket, 0.0) + count
    return vec


def _age_s(entry: FlashEntry, now: float) -> float:
    try:
        created = datetime.fromisoformat(entry.updated_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return max(0.0, now - created.timestamp())
    except ValueError:
        return 0.0


@dataclass
class Retrieval:
    """One ranked hit with its explainable score."""

    entry: FlashEntry
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        return "stale" in self.reasons


def score_entry(
    entry: FlashEntry,
    query: str,
    *,
    now: float | None = None,
    half_life_s: int = DEFAULT_HALF_LIFE_S,
) -> tuple[float, list[str]]:
    """(score, reasons) for one entry against a query — pure and deterministic."""
    now = time.time() if now is None else now
    tokens = tokenize(query)
    if not tokens:
        return 0.0, ["empty-query"]
    qvec = hash_vec(tokens)
    cosine = _cosine(qvec, _entry_vec(entry))
    if cosine <= 0.0:
        return 0.0, []
    reasons: list[str] = []
    query_set = set(tokens)
    entry_words = set(tokenize(entry.text))
    if query_set & entry_words:
        reasons.append("term-match")
    tag_words = set(tokenize(" ".join(entry.tags)))
    if query_set & tag_words:
        reasons.append("tag-match")
    task_words = set(tokenize(entry.task))
    if query_set & task_words:
        reasons.append("task-match")

    age = _age_s(entry, now)
    freshness = 0.5 ** (age / half_life_s)
    score = cosine * (0.4 + 0.6 * freshness)
    if entry.stale_after_s is not None and age > entry.stale_after_s:
        score *= 0.5
        reasons.append("stale")
    if entry.verified:
        reasons.append("verified")
        score *= 1.1
    if entry.conflicts:
        reasons.append("has-conflicts")
    if entry.private:
        reasons.append("private")
    return score, reasons


def retrieve(
    store: FlashStore,
    query: str,
    *,
    k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
    now: float | None = None,
) -> list[Retrieval]:
    """Rank the store against *query*; only entries above *min_score* return."""
    if k <= 0:
        raise ValueError("k must be positive")
    now = time.time() if now is None else now
    scored: list[Retrieval] = []
    for entry in store.all():
        score, reasons = score_entry(entry, query, now=now)
        if score >= min_score:
            scored.append(Retrieval(entry=entry, score=score, reasons=reasons))
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:k]


def context_block(retrievals: list[Retrieval], *, max_chars: int = 6000) -> str:
    """A compact, provenance-carrying block for a model prompt.

    Each entry renders its sources, uncertainty, and conflicts inline; the
    block is truncated to *max_chars* from the top hits, so unrelated or
    low-value material never reaches the model's context.
    """
    lines: list[str] = ["# Flash context (prior research, retrieved for this task)"]
    used = 0
    for hit in retrievals:
        entry = hit.entry
        header = (
            f"## [{entry.id[:8]}] {entry.text[:200]}"
            f"  (score={hit.score:.3f}; {', '.join(hit.reasons) or 'match'})"
        )
        body = [header, f"task: {entry.task or '(untagged)'}"]
        if entry.tags:
            body.append(f"tags: {', '.join(entry.tags)}")
        body.append(f"finding: {entry.text}")
        if entry.sources:
            sources = ", ".join(f"{s.title or s.url} <{s.url}>" for s in entry.sources[:5])
            body.append(f"sources: {sources}")
        if entry.uncertainty:
            body.append(f"uncertainty: {entry.uncertainty}")
        if entry.conflicts:
            body.append("conflicting evidence:")
            body.extend(f"- {c}" for c in entry.conflicts)
        if not entry.verified:
            body.append("verified: no (heuristic retrieval, not a sealed receipt)")
        if entry.private:
            body.append("PRIVATE — do not share or use in web-search queries.")
        block = "\n".join(body)
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n\n".join(lines)
