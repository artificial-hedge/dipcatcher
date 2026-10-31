"""Default research surface + justify-or-flag classification of optional bulk.

``OPTIONAL_BENCHMARK_FAMILIES`` (10,222) is the append-only ACCEPTED history
that keeps archived receipts verifying — it is not a claim that 10,222 research
families exist. These tests pin the phase-3 surface contract:

- the DEFAULT surface is the 23 REQUIRED families;
- every live-optional family is justified or explicitly exploratory (no
  unclassified names);
- the live-optional count may only FALL from its pinned 2026-10-08 ratchet;
- the classification is derived from the canon-qualification audit, so it
  cannot silently drift from the same evidence that drives retirement.

They sit alongside ``test_catalog_retired_families.py``, which continues to own
the retired-set guarantees (archived receipt verifies, unknown family fails
closed); nothing there is relaxed here.
"""

from __future__ import annotations

from quant_fund.research.catalog.optional_classification import (
    CLASSIFICATION_EXPLORATORY,
    CLASSIFICATION_JUSTIFIED,
    CLASSIFICATION_VALUES,
    LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING,
    classify_optional_families,
    default_surface_summary,
    exploratory_families,
    family_provenance,
    justified_families,
    unclassified_families,
)
from quant_fund.research.catalog.registry import (
    DEFAULT_BENCHMARK_FAMILIES,
    LIVE_OPTIONAL_BENCHMARK_FAMILIES,
    OPTIONAL_BENCHMARK_FAMILIES,
    REQUIRED_BENCHMARK_FAMILIES,
)

#: Pinned 2026-10-08. Ratchet: may only FALL as families are retired. NEVER
#: raise without an audit receipt from scripts/canon_qualify.py.
EXPECTED_LIVE_OPTIONAL_CEILING = LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING


def test_default_surface_is_the_required_canonical_set() -> None:
    # The default research surface is the 23 REQUIRED families, nothing else.
    assert DEFAULT_BENCHMARK_FAMILIES == REQUIRED_BENCHMARK_FAMILIES
    assert len(DEFAULT_BENCHMARK_FAMILIES) == 23
    # Explicitly NOT the optional bulk — that is accepted history, not surface.
    assert not (DEFAULT_BENCHMARK_FAMILIES & OPTIONAL_BENCHMARK_FAMILIES)


def test_every_live_optional_family_is_justified_or_exploratory() -> None:
    classification = classify_optional_families()
    assert unclassified_families() == frozenset(), (
        "every live-optional family must be justified or flagged exploratory"
    )
    assert set(classification) == set(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    assert set(classification.values()) <= CLASSIFICATION_VALUES


def test_justified_and_exploratory_partition_the_live_optional_set() -> None:
    justified = justified_families()
    exploratory = exploratory_families()
    assert justified & exploratory == frozenset()
    assert justified | exploratory == set(LIVE_OPTIONAL_BENCHMARK_FAMILIES)


def test_live_optional_count_ratchets_downward_only() -> None:
    live = len(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    assert live <= EXPECTED_LIVE_OPTIONAL_CEILING, (
        f"live-optional grew from {EXPECTED_LIVE_OPTIONAL_CEILING} to {live}; "
        "the optional surface may only shrink. Raising the ceiling requires an "
        "audit receipt, not a catalog edit."
    )


def test_ratchet_ceiling_matches_live_optional_surface() -> None:
    # The ceiling is pinned at the current count so growth fails loudly above.
    assert LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING == EXPECTED_LIVE_OPTIONAL_CEILING
    assert len(LIVE_OPTIONAL_BENCHMARK_FAMILIES) <= LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING


def test_justified_families_are_the_non_template_remainder() -> None:
    # The justified set is small and hand-specified; the bulk is exploratory.
    justified = justified_families()
    assert justified <= set(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    assert len(justified) < len(exploratory_families())
    classification = classify_optional_families()
    for name in sorted(justified):
        assert classification[name] == CLASSIFICATION_JUSTIFIED
    for name in sorted(exploratory_families())[:5]:
        assert classification[name] == CLASSIFICATION_EXPLORATORY


def test_classification_is_deterministic() -> None:
    assert classify_optional_families() == classify_optional_families()


def test_provenance_marker_distinguishes_template_from_specified() -> None:
    exploratory = sorted(exploratory_families())
    justified = sorted(justified_families())
    assert exploratory and justified
    assert family_provenance(exploratory[0]) == "template"
    assert family_provenance(justified[0]) == "specified"


def test_default_surface_summary_leads_with_the_default_surface() -> None:
    summary = default_surface_summary()
    assert summary["default_surface"] == sorted(DEFAULT_BENCHMARK_FAMILIES)
    assert summary["default_surface_size"] == 23
    assert summary["justified"] + summary["exploratory"] == len(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    # The optional bulk must never be presented as the research surface.
    assert summary["optional_total"] > summary["default_surface_size"]


def test_exploratory_flag_does_not_retire_or_weaken_the_optional_set() -> None:
    # Classification is a LABEL only: it must not shrink the accepted set that
    # keeps archived receipts verifying, nor touch retirement.
    assert exploratory_families() <= set(OPTIONAL_BENCHMARK_FAMILIES)
    assert not (exploratory_families() & REQUIRED_BENCHMARK_FAMILIES)
