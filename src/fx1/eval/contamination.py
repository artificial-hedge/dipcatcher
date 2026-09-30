"""Contamination audit — the section no model card has ever carried.

Three complementary probes, each with stated limitations (no single method is
sufficient; the audit reports all three):

1. **n-gram containment** — corpus examples overlapping eval prompts above a
   shingle-containment threshold. Limitation: misses paraphrases.
2. **Min-K% Prob-style membership signal** — token-probability statistics per
   eval item; contaminated items show systematically higher minimum-token
   likelihood. Computed from caller-supplied logprobs (the harness does not
   fabricate probabilities); limitation: dataset-level indicator only.
3. **Rephrased-gap (ConStat-style)** — performance delta between canonical
   eval items and meaning-preserving rephrasings; memorized items degrade
   sharply under rephrasing. Limitation: requires curated rephrasings.

The audit output is a machine-checkable report bound by hash to the corpus
and eval bank it inspected — publishable in the model card's contamination
section.
"""

from __future__ import annotations

import hashlib
import math
import statistics

from pydantic import BaseModel, Field

from fx1.data.quality import _shingles  # shared shingle implementation


class NGramFinding(BaseModel):
    example_index: int
    containment: float
    eval_index: int | None = None


class ProbeResult(BaseModel):
    method: str
    value: float | None
    threshold: float
    flagged: bool
    limitation: str


class ContaminationReport(BaseModel):
    corpus_sha256: str = Field(min_length=64, max_length=64)
    eval_bank_sha256: str = Field(min_length=64, max_length=64)
    ngram_hits: list[NGramFinding] = Field(default_factory=list)
    probes: list[ProbeResult] = Field(default_factory=list)
    overall_flagged: bool


def _hash_texts(texts: list[str]) -> str:
    """Order-independent multiset digest — per-item digests, then sorted.

    Joining raw texts on ``\n`` is ambiguous (``["a\nb"]`` and
    ``["a", "b"]`` collide); binding per-item sha256s does not leak
    corpus content and cannot be confused by embedded newlines.
    """
    inner = sorted(hashlib.sha256(t.encode()).digest() for t in texts)
    return hashlib.sha256(b"".join(inner)).hexdigest()


def ngram_containment_scan(
    corpus_texts: list[str],
    eval_prompts: list[str],
    *,
    threshold: float = 0.3,
) -> list[NGramFinding]:
    """Flag corpus examples that embed an eval prompt.

    Containment is measured per eval item — ``|doc ∩ eval_i| / |eval_i|``
    — not against the pooled union of all prompts. Pooling under-detects
    the dominant contamination case (a long training doc embedding one
    verbatim eval item scores near zero on a doc-side denominator) and
    over-flags benign docs sharing generic shingles across unrelated
    prompts. The max per-item containment is reported with the matched
    item index.
    """
    eval_sets: list[set[str]] = [_shingles(p) for p in eval_prompts]
    findings: list[NGramFinding] = []
    for i, text in enumerate(corpus_texts):
        shingles = _shingles(text)
        if not shingles:
            continue
        best, best_j = 0.0, None
        for j, item in enumerate(eval_sets):
            if not item:
                continue
            c = len(shingles & item) / len(item)
            if c > best:
                best, best_j = c, j
        if best >= threshold:
            findings.append(NGramFinding(example_index=i, containment=best, eval_index=best_j))
    return findings


def min_k_percent_probe(
    item_logprobs: list[list[float]], *, k_percent: float = 20.0
) -> ProbeResult:
    """Min-K% membership signal: mean of per-item k%-lowest token logprobs.

    Higher (less negative) values suggest memorization. Reported as a
    distributional statistic with z-score against a caller-provided reference
    mean/std if available — here we report the raw statistic and flag on a
    conservative absolute threshold only when a reference is given.
    """
    if not item_logprobs:
        return ProbeResult(
            method="min_k_percent",
            value=None,
            threshold=float("nan"),
            flagged=False,
            limitation="no logprobs supplied; probe inert (dataset-level "
            "indicator only even when active)",
        )
    stats: list[float] = []
    for logprobs in item_logprobs:
        if not logprobs:
            continue
        k = max(1, math.ceil(len(logprobs) * k_percent / 100.0))
        ordered = sorted(logprobs)
        stats.append(sum(ordered[:k]) / len(ordered[:k]))
    value = statistics.fmean(stats) if stats else float("nan")
    return ProbeResult(
        method="min_k_percent",
        value=value,
        threshold=float("nan"),
        flagged=False,
        limitation="dataset-level indicator; requires a clean-reference "
        "distribution for calibrated flagging",
    )


def rephrased_gap_probe(
    canonical_pass: list[bool], rephrased_pass: list[bool], *, budget: float = 0.3
) -> ProbeResult:
    """ConStat-style gap: memorized items degrade under rephrasing."""
    if len(canonical_pass) != len(rephrased_pass) or not canonical_pass:
        raise ValueError("paired canonical/rephrased outcomes required")
    gap = (sum(canonical_pass) - sum(rephrased_pass)) / len(canonical_pass)
    return ProbeResult(
        method="rephrased_gap",
        value=gap,
        threshold=budget,
        flagged=gap > budget,
        limitation="requires curated meaning-preserving rephrasings; gap can "
        "also reflect genuinely brittle reasoning",
    )


def run_contamination_audit(
    corpus_texts: list[str],
    eval_prompts: list[str],
    *,
    item_logprobs: list[list[float]] | None = None,
    canonical_pass: list[bool] | None = None,
    rephrased_pass: list[bool] | None = None,
) -> ContaminationReport:
    """Run all probes and emit the hash-bound, publishable report.

    Refuses empty inputs: an audit over no corpus or no eval prompts would
    certify vacuity as cleanliness — the report must never issue from
    nothing inspected.
    """
    if not corpus_texts or not eval_prompts:
        raise ValueError(
            "contamination audit requires non-empty corpus and eval prompts; "
            "an empty input would produce a clean bill of health over nothing"
        )
    hits = ngram_containment_scan(corpus_texts, eval_prompts)
    probes = [min_k_percent_probe(item_logprobs or [])]
    if canonical_pass is not None and rephrased_pass is not None:
        probes.append(rephrased_gap_probe(canonical_pass, rephrased_pass))
    ngram_probe = ProbeResult(
        method="ngram_containment",
        value=float(len(hits)),
        threshold=0.0,
        flagged=bool(hits),
        limitation="misses paraphrased or reordered contamination",
    )
    probes.insert(0, ngram_probe)
    return ContaminationReport(
        corpus_sha256=_hash_texts(corpus_texts),
        eval_bank_sha256=_hash_texts(eval_prompts),
        ngram_hits=hits,
        probes=probes,
        overall_flagged=any(p.flagged for p in probes),
    )
