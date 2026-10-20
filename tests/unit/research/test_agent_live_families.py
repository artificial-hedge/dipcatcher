"""Runtime family de-emission: RETIRED optional families never emitted.

Synthetic catalog states via monkeypatch — correctness tests only. They
assert emission policy, never market evidence.
"""

from __future__ import annotations

from typing import Any

import pytest

from quant_fund.research import agent
from quant_fund.research.catalog import registry
from quant_fund.research.catalog.optional_classification import (
    exploratory_families,
    justified_families,
)

_REQUIRED = frozenset(registry.REQUIRED_BENCHMARK_FAMILIES)


def _families(*optional: str) -> dict[str, Any]:
    return {
        **{name: {"blob": name} for name in _REQUIRED},
        **{name: {"blob": name} for name in optional},
    }


def _set_catalog(monkeypatch, optional, retired=None, live=None) -> None:
    monkeypatch.setattr(registry, "OPTIONAL_BENCHMARK_FAMILIES", frozenset(optional), raising=False)
    if retired is None:
        monkeypatch.delattr(registry, "RETIRED_BENCHMARK_FAMILIES", raising=False)
    else:
        monkeypatch.setattr(
            registry, "RETIRED_BENCHMARK_FAMILIES", frozenset(retired), raising=False
        )
    if live is None:
        monkeypatch.delattr(registry, "LIVE_OPTIONAL_BENCHMARK_FAMILIES", raising=False)
    else:
        monkeypatch.setattr(
            registry, "LIVE_OPTIONAL_BENCHMARK_FAMILIES", frozenset(live), raising=False
        )


def test_required_families_still_fully_emitted(monkeypatch):
    _set_catalog(monkeypatch, optional=["opt_live", "opt_retired"], retired=["opt_retired"])
    emitted = agent._emit_live_families(_families("opt_live", "opt_retired"))
    assert set(emitted) >= _REQUIRED


def test_no_retired_family_is_emitted(monkeypatch):
    _set_catalog(monkeypatch, optional=["opt_live", "opt_retired"], retired=["opt_retired"])
    emitted = agent._emit_live_families(_families("opt_live", "opt_retired"))
    assert "opt_retired" not in emitted
    assert "opt_live" in emitted
    assert "opt_retired" not in agent._benchmark_scorecard(emitted)


def test_live_symbol_wins_when_catalog_has_landed_it(monkeypatch):
    _set_catalog(monkeypatch, optional=["opt_live", "opt_retired"], live=["opt_live"])
    emitted = agent._emit_live_families(_families("opt_live", "opt_retired"))
    assert set(emitted) == _REQUIRED | {"opt_live"}
    assert agent._retired_optional_families() == frozenset({"opt_retired"})


def test_unknown_non_family_keys_pass_through_unchanged(monkeypatch):
    _set_catalog(monkeypatch, optional=["opt_retired"], retired=["opt_retired"])
    emitted = agent._emit_live_families(_families("opt_retired") | {"misc_blob": {}})
    assert "misc_blob" in emitted


def test_current_catalog_state_emits_everything_unless_retired_lands():
    families = _families()
    assert set(agent._emit_live_families(families)) >= _REQUIRED
    if hasattr(registry, "RETIRED_BENCHMARK_FAMILIES") or hasattr(
        registry, "LIVE_OPTIONAL_BENCHMARK_FAMILIES"
    ):
        retired = agent._retired_optional_families()
        assert not (set(families) & retired)


def test_audited_template_provenance_flows_into_live_blob_without_mutating_bench() -> None:
    template = sorted(exploratory_families())[0]
    specified = sorted(justified_families())[0]
    source = _families(template, specified)
    emitted = agent._emit_live_families(source)

    assert source[template] == {"blob": template}
    assert emitted[template] == {"blob": template, "provenance": "template"}
    assert emitted[specified] == source[specified]
    assert "provenance" not in emitted[specified]


def test_template_provenance_does_not_turn_empty_or_nan_blob_into_observation() -> None:
    template = sorted(exploratory_families())[0]
    emitted = agent._emit_live_families({template: {}})
    assert emitted[template] == {"provenance": "template"}
    empty_card = agent._benchmark_scorecard(emitted)[template]
    assert empty_card["executed"] is False
    assert empty_card["nonempty"] is False
    assert empty_card["finite_observation"] is False

    nan_blob = agent._emit_live_families({template: {"metric": float("nan")}})
    nan_card = agent._benchmark_scorecard(nan_blob)[template]
    assert nan_card["finite_observation"] is False


def test_template_provenance_conflict_fails_closed() -> None:
    template = sorted(exploratory_families())[0]
    with pytest.raises(ValueError, match="conflicting provenance"):
        agent._emit_live_families({template: {"provenance": "specified"}})
    with pytest.raises(TypeError, match="must emit a dict"):
        agent._emit_live_families({template: None})
