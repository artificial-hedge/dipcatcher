"""Collectability placeholder — REAL CONTENT IS OWNED BY ``canon-integrity``.

WHY THIS FILE EXISTS
--------------------
``catalog/registry.py`` imports ``RETIRED_BENCHMARK_FAMILIES`` from this module.
An in-flight edit added that import line *before* this module was written, which
made ``quant_fund.research.catalog`` un-importable at all::

    ModuleNotFoundError: No module named
    'quant_fund.research.catalog.retired_families'

Because ``registry.py`` is imported by ``catalog/__init__.py``, that single
missing module broke collection for the ENTIRE test suite (pytest exit 4) and
caused cascading xdist worker crashes. This placeholder restores
collectability only.

THIS IS DELIBERATELY NOT THE ANSWER
-----------------------------------
``RETIRED_BENCHMARK_FAMILIES`` is EMPTY here. That is wrong on purpose. The
retirement set MUST be derived from ``quality/canon_qualification_audit.json``
(the deterministic corpus-qualification classifier output) so that the audit
and the registry cannot drift apart. Hand-writing or stubbing that set would
make the runtime de-emission filter in ``research/agent.py`` a silent no-op --
exactly the kind of quiet non-function this repository's honesty contract
exists to prevent.

``canon-integrity`` overwrites this file with the real derivation. Until that
lands, ``tests/unit/research/test_catalog_retired_families.py`` is EXPECTED to
fail on the derivation-dependent assertions (``test_retired_set_matches_
qualification_audit``, ``test_archived_receipt_with_retired_family_still_
verifies``). Those failures are the correct signal and are NOT a defect in this
placeholder -- do not "fix" them by weakening the tests.

REQUIRED SHAPE (see tests/unit/research/test_catalog_retired_families.py)
------------------------------------------------------------------------
``RETIRED_BENCHMARK_FAMILIES`` maps a retired family name to an entry carrying
at least:

- ``reason``: a specific, non-vague explanation (>= 30 chars) that states the
  template evidence -- the tests assert ``"template"`` appears in it.
- ``schema_version``: one of ``RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED``.

Every retired name must live inside ``OPTIONAL_BENCHMARK_FAMILIES`` and outside
``REQUIRED_BENCHMARK_FAMILIES``. ``OPTIONAL_BENCHMARK_FAMILIES`` stays
append-only and UNCHANGED: it is the historical ACCEPTED set and old receipts
must keep verifying.
"""

from __future__ import annotations

from typing import Final

# Placeholder only -- canon-integrity replaces this with the derived set.
RETIRED_BENCHMARK_FAMILIES: Final[dict[str, dict[str, object]]] = {}
