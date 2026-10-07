"""Runtime family de-emission: RETIRED optional families never emitted.

Synthetic catalog states via monkeypatch — correctness tests only. They
assert emission policy, never market evidence, and carry no live-trading or
profitability claim.
"""

from __future__ import annotations

from typing import Any

from quant_fund.research import agent
from quant_fund.research.catalog import registry

_REQUIRED = frozenset(registry.REQUIRED_BENCHMARK_FAMILIES)


def _families(*optional: str) -> dict[str, Any]:
    return {**{name: {"blob": name} for name in _REQUIRED}, **{name: {"blob": name} for name in optional}}


def _set_catalog(monkeypatch, optional, retired=None, live=None) -> None:
    monkeypatch.setattr(registry, "OPTIONAL_BENCHMARK_FAMILIES", frozenset(optional), raising=False)
    if retired is None:
        monkeypatch.delattr(registry, "RETIRED_BENCHMARK_FAMILIES", raising=False)
    else:
        monkeypatch.setattr(registry, "RETIRED_BENCHMARK_FAMILIES", frozenset(retired), raising=False)
    if live is None:
        monkeypatch.delattr(registry, "LIVE_OPTIONAL_BENCHMARK_FAMILIES", raising=False)
    else:
        monkeypatch.setattr(registry, "LIVE_OPTIONAL_BENCHMARK_FAMILIES", frozenset(live), raising=False)


def test_required_families_still_fully_emitted(monkeypatch):
    _set_catalog(monkeypatch, optional=["opt_live", "opt_retired"], retired=["opt_retired"])
    emitted = agent._emit_live_families(_families("opt_live", "opt_retired"))
    assert set(emitted) >= _REQUIRED


def test_no_retired_family_is_emitted(monkeypatch):
    _set_catalog(monkeypatch, optional=["opt_live", "opt_retired"], retired=["opt_retired"])
    emitted = agent._emit_live_families(_families("opt_live", "opt_retired"))
    assert "opt_retired" not in emitted
    assert "opt_live" in emitted


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
