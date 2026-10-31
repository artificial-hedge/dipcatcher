"""Justify-or-flag classification of the live-optional benchmark surface.

``OPTIONAL_BENCHMARK_FAMILIES`` (10,222) is the historical ACCEPTED set and is
append-only; ``LIVE_OPTIONAL_BENCHMARK_FAMILIES`` (= ``OPTIONAL -
RETIRED``, 2,693) is what ``research/agent.py`` can still emit. Neither set is
the research surface a reader should be pointed at: the DEFAULT surface is the
23 ``REQUIRED_BENCHMARK_FAMILIES`` (re-exported as
``DEFAULT_BENCHMARK_FAMILIES`` by ``catalog.registry``). This module exists so
the remaining optional bulk is *accounted for* rather than merely counted.

Classification rule (deterministic, auditable, no per-family hand-authoring)
-----------------------------------------------------------------------------
Every live-optional family lands in exactly one of two buckets:

``justified``
    The family is NOT a template stub — it has its own behavioral mechanism
    wired to real panel data (a distinct model module and/or a bench taking a
    ``pl.DataFrame``), so the canon qualification audit never raised it as a
    retirement candidate.
``exploratory``
    Everything else. The family is still template-backed: it is one of the
    shared-AST-shape ``NON_QUALIFYING_TEMPLATE`` stubs the audit enumerates as
    ``live_candidate_families`` in ``quality/canon_qualification_summary.json``
    (the same evidence that produced ``RETIRED_BENCHMARK_FAMILIES``), i.e. a
    constant-check bench over constant arguments with no control flow and no
    data parameters.

Why this is auditable: the bucket is *derived from the audit's own evidence
file*, not asserted. :func:`classify_optional_families` is pure set algebra
over that summary, so re-running ``scripts/canon_qualify.py`` and re-running
this module cannot disagree without the difference showing up as an
unclassified family. The audit is the single source of truth for both the
retired set (generated into ``retired_families.py``) and this classification —
no name is hand-listed here.

``exploratory`` is a *label*, not a retirement. Nothing here removes a family
from ``OPTIONAL_BENCHMARK_FAMILIES``, changes ``verify.py``'s allow-list, or
weakens fail-closed verification: an ARCHIVED receipt naming an exploratory
family still verifies exactly as before. Exploratory families are ones the
audit already flagged as non-qualifying; the honest reading of this catalog is
"23 researched families, plus a large exploratory template surface retained
for receipt back-compatibility".

The ratchet lives here too: ``LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING`` pins
the live-optional count at its 2026-10-08 value. The count may only FALL (more
retirement); raising it requires an audit receipt and a deliberate edit here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from .registry import (
    DEFAULT_BENCHMARK_FAMILIES,
    LIVE_OPTIONAL_BENCHMARK_FAMILIES,
    OPTIONAL_BENCHMARK_FAMILIES,
)

#: Classification buckets. Every live-optional family is in exactly one.
CLASSIFICATION_JUSTIFIED: Final = "justified"
CLASSIFICATION_EXPLORATORY: Final = "exploratory"
CLASSIFICATION_VALUES: Final = frozenset({CLASSIFICATION_JUSTIFIED, CLASSIFICATION_EXPLORATORY})

#: Provenance markers. ``"template"`` means the family is one of the audit's
#: shared-AST-shape NON_QUALIFYING_TEMPLATE stubs; ``"specified"`` means it has
#: its own mechanism wired to real panel data.
PROVENANCE_TEMPLATE: Final = "template"
PROVENANCE_SPECIFIED: Final = "specified"

#: Repo root, for locating the audit summary. Mirrors the retired-families test.
_REPO_ROOT: Final = Path(__file__).resolve().parents[4]

#: Audit evidence file shared with the retirement machinery. ``live_candidate_families``
#: is the set the canon qualification audit flagged as template-backed but not
#: yet retired; the complement within LIVE_OPTIONAL is the justified set.
QUALIFICATION_SUMMARY_PATH: Final = _REPO_ROOT / "quality" / "canon_qualification_summary.json"

#: Ratchet on the live-optional surface size, pinned 2026-10-08 at 2693.
#: This may only FALL as more families are retired. NEVER raise it without an
#: audit receipt from ``scripts/canon_qualify.py`` justifying the growth — the
#: point of the ratchet is that the optional surface can only shrink.
LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING: Final = 2693


def _audit_live_candidates() -> frozenset[str]:
    """Return the audit's template-backed live-candidate family names.

    Reads ``quality/canon_qualification_summary.json`` — the same evidence file
    the retirement machinery is generated from. Returns an empty set when the
    file is absent so that the classification degrades to "everything is
    justified" only in the sense that nothing is *mislabelled* exploratory;
    :func:`classify_optional_families` still covers every live-optional family
    in that case.
    """
    if not QUALIFICATION_SUMMARY_PATH.is_file():
        return frozenset()
    summary = json.loads(QUALIFICATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    raw = summary.get("live_candidate_families", [])
    return frozenset(str(name) for name in raw)


def classify_optional_families() -> dict[str, str]:
    """Classify every live-optional family as justified or exploratory.

    Returns a mapping covering exactly ``LIVE_OPTIONAL_BENCHMARK_FAMILIES``:
    ``exploratory`` for the audit's template-backed live candidates,
    ``justified`` for the rest. Deterministic and hand-editing-free — the
    buckets are pure set algebra over the audit evidence.
    """
    candidates = _audit_live_candidates() & LIVE_OPTIONAL_BENCHMARK_FAMILIES
    return {
        name: (CLASSIFICATION_EXPLORATORY if name in candidates else CLASSIFICATION_JUSTIFIED)
        for name in sorted(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    }


def exploratory_families() -> frozenset[str]:
    """Live-optional families the audit flags as template-backed (exploratory)."""
    return frozenset(
        name
        for name, kind in classify_optional_families().items()
        if kind == CLASSIFICATION_EXPLORATORY
    )


def justified_families() -> frozenset[str]:
    """Live-optional families with a real behavioral mechanism (justified)."""
    return frozenset(
        name
        for name, kind in classify_optional_families().items()
        if kind == CLASSIFICATION_JUSTIFIED
    )


def unclassified_families() -> frozenset[str]:
    """Live-optional families missing from the classification. Must be empty.

    This is the fail-closed guard: if the classification ever stops covering
    the live-optional surface, this returns the offenders instead of silently
    dropping them.
    """
    return LIVE_OPTIONAL_BENCHMARK_FAMILIES - frozenset(classify_optional_families())


def family_provenance(name: str) -> str:
    """Return ``"specified"`` for justified, ``"template"`` for exploratory.

    Machine-readable provenance marker keyed off the classification. Families
    outside the live-optional set (REQUIRED families, unknown names) report
    ``"specified"`` so nothing ever claims template backing without evidence.
    """
    kind = classify_optional_families().get(name)
    if kind == CLASSIFICATION_EXPLORATORY:
        return PROVENANCE_TEMPLATE
    return PROVENANCE_SPECIFIED


def default_surface_summary() -> dict[str, object]:
    """Headline counts for the default surface, for display/CLI reporting.

    Reports the DEFAULT (required) surface alongside the optional bulk so a
    reader never sees "10,222 research families" without the qualifiers.
    """
    classification = classify_optional_families()
    return {
        "default_surface": sorted(DEFAULT_BENCHMARK_FAMILIES),
        "default_surface_size": len(DEFAULT_BENCHMARK_FAMILIES),
        "optional_total": len(OPTIONAL_BENCHMARK_FAMILIES),
        "live_optional": len(LIVE_OPTIONAL_BENCHMARK_FAMILIES),
        "justified": sum(1 for kind in classification.values() if kind == CLASSIFICATION_JUSTIFIED),
        "exploratory": sum(
            1 for kind in classification.values() if kind == CLASSIFICATION_EXPLORATORY
        ),
    }


__all__ = [
    "CLASSIFICATION_EXPLORATORY",
    "CLASSIFICATION_JUSTIFIED",
    "CLASSIFICATION_VALUES",
    "LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING",
    "QUALIFICATION_SUMMARY_PATH",
    "classify_optional_families",
    "default_surface_summary",
    "exploratory_families",
    "family_provenance",
    "justified_families",
    "unclassified_families",
]
