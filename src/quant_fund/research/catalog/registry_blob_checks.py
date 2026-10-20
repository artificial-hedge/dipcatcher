"""Research family blob honesty helpers (forbidden keys, observations)."""

from __future__ import annotations

import math

# Headline P&L / ratio key *tokens* that must never appear in research family /
# scorecard blobs (not the paper analytics_export schema — that catalog allows
# equity/stress pnl/nav diagnostics under live_pnl_claim=false).
# Matching is by underscore-token on mapping keys (case-insensitive), so
# ``flag_high_sharpe`` / ``shock_down_pnl`` / ``calmar_diagnostic`` all fail closed.
FORBIDDEN_RESEARCH_METRIC_KEYS = frozenset(
    {
        "sharpe",
        "sortino",
        "calmar",
        "pnl",
        "nav",
    }
)


def family_blob_forbidden_metrics_absent(payload: object) -> bool:
    """Return True iff *payload* has no forbidden research-headline metric keys.

    Fail closed: any mapping key whose underscore tokens include sharpe / sortino /
    calmar / pnl / nav marks the blob unclean. Scalar metric values are not scanned.

    Scope: research family / scorecard blobs only. Paper ``analytics_export`` may
    contain equity ``nav_*`` / stress ``*_pnl`` diagnostics; validate those with
    ``validate_analytics_export`` (live_pnl_claim fail-closed), not this helper.

    ``live_pnl_claim`` itself is exempt at any depth only when its value is
    literally False: it is the honesty flag, not a metric. Receipts that embed
    other receipts carry it nested (e.g. a tournament manifest quoting its
    benchmark manifest); a true or malformed nested flag must fail closed too.
    """
    from quant_fund.utils.forbidden_research_keys import mapping_key_forbidden

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "live_pnl_claim":
                if value is not False:
                    return False
                continue
            if mapping_key_forbidden(str(key), FORBIDDEN_RESEARCH_METRIC_KEYS):
                return False
            if not family_blob_forbidden_metrics_absent(value):
                return False
    elif isinstance(payload, (list, tuple)):
        return all(family_blob_forbidden_metrics_absent(item) for item in payload)
    return True


def _has_template_provenance(payload: dict[object, object]) -> bool:
    marker = payload.get("provenance")
    return isinstance(marker, str) and marker == "template"


def family_blob_executed(payload: object) -> bool:
    """An exact template label alone does not count as an executed blob."""
    if isinstance(payload, dict) and _has_template_provenance(payload):
        return any(key != "provenance" for key in payload)
    return bool(payload)


def family_blob_nonempty(payload: object) -> bool:
    """An exact template label alone does not count as a nonempty blob."""
    return family_blob_executed(payload)


def family_blob_has_finite_observation(payload: object) -> bool:
    """True iff *payload* recursively contains at least one finite observation.

    Lifted from research.agent ``_benchmark_scorecard`` nested ``has_observation``
    (Day Wave 44). Semantics:
    - dict / list / tuple → any child is an observation, excluding an exact
      ``provenance: template`` label
    - bool → True
    - int (incl. numpy integer scalars via ``.item()``) → True
    - float (incl. numpy floating) → ``math.isfinite``
    - str → nonempty strip and not ``nan`` / ``none`` (case-insensitive)
    - else → False

    Empty ``{}``, template-label-only, and all-NaN blobs → False. Used by soft
    scorecard forge verify.
    """
    if isinstance(payload, dict):
        template_labeled = _has_template_provenance(payload)
        return any(
            family_blob_has_finite_observation(item)
            for key, item in payload.items()
            if not (template_labeled and key == "provenance")
        )
    if isinstance(payload, (list, tuple)):
        return any(family_blob_has_finite_observation(item) for item in payload)
    if isinstance(payload, bool):
        return True
    if isinstance(payload, int):
        return True
    if isinstance(payload, float):
        return math.isfinite(payload)
    # Numpy scalars are not always subclasses of int/float — recurse via .item().
    if (
        hasattr(payload, "dtype")
        and hasattr(payload, "item")
        and not isinstance(payload, (bytes, bytearray, memoryview))
    ):
        try:
            return family_blob_has_finite_observation(payload.item())
        except (ValueError, TypeError, AttributeError):
            return False
    return (
        isinstance(payload, str)
        and bool(payload.strip())
        and payload.lower() not in {"nan", "none"}
    )
