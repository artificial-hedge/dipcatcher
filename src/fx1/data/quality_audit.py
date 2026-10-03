"""quality_audit — adversarial probes on the corpus quality gates.

Pinned contract for ``dedup_and_filter``:

- Exact duplicates (sha256 of concatenated message text) are removed.
- Eval contamination uses **per-item** containment — a corpus example
  embedding ≥60% of one eval prompt's shingles flags; fragments across
  unrelated prompts do not sum to a hit.
- NFKC folding closes the formatting-evasion class (fullwidth /
  punctuation variants tokenize identically).
- Near-duplicates are Jaccard ≥0.9 against every kept item
  sharing a shingle (inverted index — no distance window).

**Pinned caveats**: overlength drops are silent — the report has no
``overlong_removed`` counter, so ``loaded`` need not equal kept+removed.
The near-dup check is NOT window-bounded: an inverted shingle index
compares each candidate against every kept example sharing a shingle, so
a duplicate any distance later is caught (``far_apart_dup_caught``).

``frozen_split``: deterministic under a fixed seed, val_fraction bounds
enforced, manifest digests bind the written files' contents.

Sealed ``quality_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["quality_audit", "quality_audit_bench"]

_EVAL = " ".join(f"evalterm{i}" for i in range(30))  # 30-token eval prompt


def _ex(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def _filler(i: int) -> dict[str, Any]:
    # unique long filler so the near-dup index sees unrelated kept items
    return _ex(" ".join(f"f{i}_{j}" for j in range(12)))


def quality_audit() -> dict[str, Any]:
    from fx1.data.quality import dedup_and_filter, frozen_split

    out: dict[str, Any] = {}

    # exact dupes
    kept, rep = dedup_and_filter(
        [_ex("same text here"), _ex("same text here"), _ex("different")],
        eval_prompts=[],
    )
    out["exact_dedup"] = len(kept) == 2 and rep.exact_duplicates_removed == 1

    # contamination: verbatim eval embed flags; unrelated fragments don't
    contaminated = _ex(f"prefix stuff {_EVAL} suffix stuff")
    benign = _ex("completely unrelated corpus text about bonds")
    kept, rep = dedup_and_filter([contaminated, benign], eval_prompts=[_EVAL])
    out["contamination_flags"] = rep.contaminated_removed == 1 and kept == [benign]

    # NFKC evasion: fullwidth eval prompt must still flag
    fullwidth = "".join(chr(ord(c) + 0xFEE0) if 33 <= ord(c) <= 126 else c for c in _EVAL)
    kept, rep = dedup_and_filter([_ex(fullwidth)], eval_prompts=[_EVAL])
    out["nfkc_evasion_closed"] = rep.contaminated_removed == 1

    # near-dup within window: 20-token base + 1 appended token →
    # 13 shingles shared of 14 union = Jaccard 0.93 ≥ 0.9
    base = _ex(" ".join(f"shared{i}" for i in range(20)))
    near = _ex(" ".join(f"shared{i}" for i in range(20)) + " tail")
    kept, rep = dedup_and_filter([base, near], eval_prompts=[])
    out["near_dup_windowed"] = rep.near_duplicates_removed == 1

    # >500 kept between the pair: the inverted shingle index compares a
    # candidate against EVERY kept example sharing a shingle — there is no
    # distance-bounded window for a near-dup to slip past.
    many = [base] + [_filler(i) for i in range(502)] + [near]
    kept, rep = dedup_and_filter(many, eval_prompts=[])
    out["far_apart_dup_caught"] = rep.near_duplicates_removed == 1 and len(kept) == 503

    # overlength drop is uncounted
    kept, rep = dedup_and_filter([_ex("x" * 40_000), benign], eval_prompts=[], max_len_chars=32_000)
    accounted = (
        rep.exact_duplicates_removed
        + rep.near_duplicates_removed
        + rep.contaminated_removed
        + rep.kept
    )
    out["overlong_silent_drop"] = rep.loaded == 2 and accounted == 1

    # frozen_split: determinism + bounds + digest binding
    exs = [_filler(i) for i in range(40)]
    with tempfile.TemporaryDirectory() as tmp:
        m1 = frozen_split(exs, Path(tmp) / "a")
        m2 = frozen_split(exs, Path(tmp) / "b")
        out["split_deterministic"] = m1 == m2
        out["split_counts"] = m1.train_count + m1.val_count == 40 and m1.val_count == 2
        # digest binds file content
        line = (Path(tmp) / "a.train.jsonl").read_text()
        import hashlib

        out["digest_binds_bytes"] = hashlib.sha256(line.encode()).hexdigest() == m1.train_sha256
        for bad in (0.0, 0.5, -0.1):
            try:
                frozen_split(exs, Path(tmp) / "x", val_fraction=bad)
                out.setdefault("bound_fails", []).append(bad)
            except ValueError:
                pass
        out["bounds_enforced"] = out.get("bound_fails", []) == []
    return out


def quality_audit_bench() -> dict[str, Any]:
    r = quality_audit()
    ok = (
        all(r[k] is True for k in r if k not in {"split_counts", "bounds_enforced", "bound_fails"})
        and r["split_counts"] is True
        and r["bounds_enforced"] is True
    )
    out: dict[str, Any] = {
        "kind": "quality_audit",
        "schema": "quality_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {
                "near_dup_window_bounded": not r["far_apart_dup_caught"],
                "overlong_drops_uncounted": r["overlong_silent_drop"],
            },
            "ok": ok,
        },
        "interpretation": (
            "Quality-gate contract holds: exact dupes removed, per-item "
            "containment catches verbatim + NFKC-folded eval embeds, "
            "frozen split deterministic with digest-bound files. Flag "
            "pinned: overlength drops are uncounted in the report. Near-dup "
            "dedup is distance-unbounded (inverted shingle index)."
            if ok
            else f"QUALITY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
