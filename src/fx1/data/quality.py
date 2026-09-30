"""Industry-grade corpus quality gates for fx-1.

No example enters training without passing: exact-duplicate removal,
near-duplicate shingle screening, length-distribution reporting,
**eval-contamination checks** (no corpus example may overlap the eval bank),
and a frozen train/validation split with a hash manifest — so every training
run can prove exactly which data it saw.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from quant_fund.utils.atomicio import atomic_write_text

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    """NFKC-folded, punctuation-free tokens.

    Formatting variation (punctuation, full-width characters, hyphenation)
    must not let a corpus copy of an eval prompt escape containment — e.g.
    "ratio, please" and "ratio; please" tokenize identically here.
    """
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def _shingles(text: str, n: int = 8) -> set[str]:
    tokens = _tokens(text)
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _text_of(example: dict[str, Any]) -> str:
    return "\n".join(str(m.get("content", "")) for m in example.get("messages", []))


class QualityReport(BaseModel):
    loaded: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    contaminated_removed: int
    over_length_removed: int = 0
    empty_removed: int = 0
    kept: int
    token_length_p50: int
    token_length_p99: int
    max_length: int


def dedup_and_filter(
    examples: list[dict[str, Any]],
    *,
    eval_prompts: list[str],
    containment_threshold: float = 0.6,
    max_len_chars: int = 32_000,
) -> tuple[list[dict[str, Any]], QualityReport]:
    """Dedup + decontaminate + length-filter. Fail-closed on eval overlap."""
    seen_exact: set[str] = set()
    kept: list[dict[str, Any]] = []
    kept_shingles: list[set[str]] = []
    # Inverted index shingle -> kept-example positions, so the near-duplicate
    # check compares a candidate against EVERY kept example that shares a
    # shingle — no distance-bounded window for a duplicate to slip past.
    shingle_to_kept: dict[str, list[int]] = {}
    # Per-item shingle sets — contamination is measured as how much of an
    # eval ITEM is embedded in the doc (|doc ∩ item| / |item|), not how much
    # of the doc overlaps the pooled union. A long doc embedding one verbatim
    # eval prompt must flag; fragments of unrelated prompts must not sum to
    # a hit.
    eval_sets: list[set[str]] = [_shingles(prompt) for prompt in eval_prompts]
    eval_sets = [s for s in eval_sets if s]
    exact_dupes = near_dupes = contaminated = over_length = empty = 0
    lengths: list[int] = []
    for example in examples:
        text = _text_of(example)
        if not text.strip():
            empty += 1
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen_exact:
            exact_dupes += 1
            continue
        shingles = _shingles(text)
        if (
            shingles
            and eval_sets
            and any(len(shingles & item) / len(item) >= containment_threshold for item in eval_sets)
        ):
            contaminated += 1
            continue
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(shingle_to_kept.get(shingle, ()))
        if any(
            len(shingles & kept_shingles[i]) / len(shingles | kept_shingles[i]) >= 0.9
            for i in candidates
        ):
            near_dupes += 1
            continue
        if len(text) > max_len_chars:
            over_length += 1
            continue
        seen_exact.add(digest)
        position = len(kept_shingles)
        kept_shingles.append(shingles)
        for shingle in shingles:
            shingle_to_kept.setdefault(shingle, []).append(position)
        lengths.append(len(text.split()))
        kept.append(example)
    lengths.sort()
    p50 = lengths[len(lengths) // 2] if lengths else 0
    p99 = lengths[min(int(len(lengths) * 0.99), len(lengths) - 1)] if lengths else 0
    report = QualityReport(
        loaded=len(examples),
        exact_duplicates_removed=exact_dupes,
        near_duplicates_removed=near_dupes,
        contaminated_removed=contaminated,
        over_length_removed=over_length,
        empty_removed=empty,
        kept=len(kept),
        token_length_p50=p50,
        token_length_p99=p99,
        max_length=lengths[-1] if lengths else 0,
    )
    return kept, report


class SplitManifest(BaseModel):
    """Frozen split manifest — training runs must cite this."""

    seed: int
    train_sha256: str = Field(min_length=64, max_length=64)
    val_sha256: str = Field(min_length=64, max_length=64)
    train_count: int
    val_count: int


def _hash_lines(lines: list[str]) -> str:
    """SHA-256 of the exact bytes the split writer emits for *lines*.

    The manifest must pin the file on disk, not an ambiguous
    concatenation — ``["ab", "c"]`` and ``["a", "bc"]`` are different
    files and must not share a digest.
    """
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def frozen_split(
    examples: list[dict[str, Any]],
    out_prefix: str | Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 17,
) -> SplitManifest:
    """Deterministically split and write train/val JSONL + manifest."""
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    cut = max(1, int(len(indices) * val_fraction)) if len(indices) > 1 else 0
    val_idx, train_idx = set(indices[:cut]), set(indices[cut:])
    if not val_idx or not train_idx:
        raise ValueError(
            "frozen split needs at least 2 examples — an empty train or val "
            "file certifies a split that never happened"
        )
    train_lines = [json.dumps(examples[i], sort_keys=True) for i in sorted(train_idx)]
    val_lines = [json.dumps(examples[i], sort_keys=True) for i in sorted(val_idx)]
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(Path(f"{prefix}.train.jsonl"), "\n".join(train_lines) + "\n")
    atomic_write_text(Path(f"{prefix}.val.jsonl"), "\n".join(val_lines) + "\n")
    manifest = SplitManifest(
        seed=seed,
        train_sha256=_hash_lines(train_lines),
        val_sha256=_hash_lines(val_lines),
        train_count=len(train_lines),
        val_count=len(val_lines),
    )
    atomic_write_text(Path(f"{prefix}.split.json"), manifest.model_dump_json(indent=2))
    return manifest
