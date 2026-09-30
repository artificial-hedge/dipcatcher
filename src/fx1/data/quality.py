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
    shingle_index: list[set[str]] = []
    # Per-item shingle sets — contamination is measured as how much of an
    # eval ITEM is embedded in the doc (|doc ∩ item| / |item|), not how much
    # of the doc overlaps the pooled union. A long doc embedding one verbatim
    # eval prompt must flag; fragments of unrelated prompts must not sum to
    # a hit.
    eval_sets: list[set[str]] = [_shingles(prompt) for prompt in eval_prompts]
    eval_sets = [s for s in eval_sets if s]
    exact_dupes = near_dupes = contaminated = 0
    lengths: list[int] = []
    for example in examples:
        text = _text_of(example)
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
        if any(
            shingles and len(shingles & prior) / max(len(shingles | prior), 1) >= 0.9
            for prior in shingle_index[-500:]
        ):
            near_dupes += 1
            continue
        if len(text) > max_len_chars:
            continue
        seen_exact.add(digest)
        shingle_index.append(shingles)
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
    return hashlib.sha256("".join(lines).encode()).hexdigest()


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
    train_lines = [json.dumps(examples[i], sort_keys=True) for i in sorted(train_idx)]
    val_lines = [json.dumps(examples[i], sort_keys=True) for i in sorted(val_idx)]
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.train.jsonl").write_text("\n".join(train_lines) + "\n")
    Path(f"{prefix}.val.jsonl").write_text("\n".join(val_lines) + "\n")
    manifest = SplitManifest(
        seed=seed,
        train_sha256=_hash_lines(train_lines),
        val_sha256=_hash_lines(val_lines),
        train_count=len(train_lines),
        val_count=len(val_lines),
    )
    Path(f"{prefix}.split.json").write_text(manifest.model_dump_json(indent=2))
    return manifest
