"""Planner-facing family catalog: parse, load, match, and fail closed."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from typing import Any

import pytest

from fx1.interactive.families import (
    DEFAULT_ALIASES,
    FamilyCatalog,
    FamilyState,
    families_prompt_block,
    load_catalog,
    resolve_families,
)

_NOTE = (
    "dipcatcher research executes the REQUIRED scorecard plus live optional families "
    "per its config; per-family execution constraint is not supported yet — family "
    "selection narrows eligibility and reporting, not the benches executed."
)
_CLOSED = (
    "family catalog unavailable — family eligibility could not be checked; "
    "no family-level claims will be made (fail-closed)"
)
_SCORECARD_NOTE = (
    "note: the research command runs its configured scorecard; family selection "
    "narrows eligibility and reporting, not the benches executed."
)


def _document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "catalog_version": 2,
        "receipt_schema_version": 2,
        "required": ["conformal", "distribution", "northset", "tail", "volatility"],
        "live_optional": ["candle_order_book", "regime"],
        "retired_count": 7529,
        "retired_sample": ["dead_bench", "old_family"],
        "order": ["volatility", "distribution", "tail", "conformal", "northset"],
        "forbidden_metric_keys": ["calmar", "nav", "pnl", "sharpe", "sortino"],
        "runner": {
            "command": "research",
            "selection": "config",
            "execution_constraint": False,
            "note": _NOTE,
        },
        "generated_at": "2026-10-12T00:00:00Z",
    }


def _resolver_catalog() -> FamilyCatalog:
    targets = set(DEFAULT_ALIASES.values())
    return FamilyCatalog(
        schema_version=1,
        catalog_version=2,
        receipt_schema_version=2,
        required=frozenset(targets - {"regime"}),
        live_optional=frozenset({"regime", "candle_order_book"}),
        retired_count=7529,
        retired_sample=frozenset({"old_family", "dead_bench"}),
        order=tuple(sorted(targets - {"regime"})),
        generated_at="2026-10-12T00:00:00Z",
        source="harness",
        execution_constraint=False,
        note="per-family execution constraint is not supported yet",
    )


class _Result:
    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _Harness:
    def __init__(self, result: _Result | None = None, exc: BaseException | None = None) -> None:
        self._result = result
        self._exc = exc
        self.calls = 0

    def run(self, name: str, extra_args: list[str] | None = None) -> _Result:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


def test_from_json_reads_a_realistic_document() -> None:
    document = _document()
    catalog = FamilyCatalog.from_json(json.dumps(document))
    assert catalog.schema_version == 1
    assert catalog.catalog_version == 2
    assert catalog.receipt_schema_version == 2
    assert catalog.required == frozenset(document["required"])
    assert catalog.live_optional == frozenset(document["live_optional"])
    assert catalog.retired_count == 7529
    assert catalog.retired_sample == frozenset(document["retired_sample"])
    assert catalog.order == tuple(document["order"])
    assert catalog.generated_at == "2026-10-12T00:00:00Z"
    assert catalog.source == "harness"
    assert catalog.execution_constraint is False
    assert catalog.note == _NOTE
    assert catalog.available


def test_from_json_rejects_unknown_schema() -> None:
    document = _document()
    document["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported family catalog schema"):
        FamilyCatalog.from_json(json.dumps(document))


def test_from_json_rejects_wrong_types() -> None:
    required_bad = _document()
    required_bad["required"] = "conformal"
    with pytest.raises(ValueError):
        FamilyCatalog.from_json(json.dumps(required_bad))
    count_bad = _document()
    count_bad["retired_count"] = "7529"
    with pytest.raises(ValueError):
        FamilyCatalog.from_json(json.dumps(count_bad))
    bool_count = _document()
    bool_count["retired_count"] = True
    with pytest.raises(ValueError):
        FamilyCatalog.from_json(json.dumps(bool_count))


def test_unavailable_is_empty_and_not_available() -> None:
    catalog = FamilyCatalog.unavailable("registry offline")
    assert catalog.source == "unavailable"
    assert catalog.note == "registry offline"
    assert catalog.required == frozenset()
    assert catalog.live_optional == frozenset()
    assert catalog.retired_sample == frozenset()
    assert catalog.retired_count == 0
    assert catalog.order == ()
    assert catalog.execution_constraint is False
    assert not catalog.available
    assert catalog.eligible() == frozenset()


def test_state_of_covers_required_optional_retired_and_unknown() -> None:
    catalog = _resolver_catalog()
    assert catalog.state_of("conformal") is FamilyState.REQUIRED
    assert catalog.state_of("regime") is FamilyState.OPTIONAL
    assert catalog.state_of("candle_order_book") is FamilyState.OPTIONAL
    assert catalog.state_of("old_family") is FamilyState.RETIRED
    assert catalog.state_of("not_a_family") is FamilyState.UNKNOWN


def test_eligible_is_required_plus_live_optional() -> None:
    catalog = _resolver_catalog()
    assert catalog.eligible() == catalog.required | catalog.live_optional
    assert catalog.retired_sample.isdisjoint(catalog.eligible())


def test_describe_available_and_unavailable() -> None:
    catalog = _resolver_catalog()
    text = catalog.describe()
    assert text.startswith("family catalog v2 (receipt schema v2): ")
    expected = (
        f"{len(catalog.required)} required · {len(catalog.live_optional)} live optional · "
        "7529 retired · source harness · note: per-family execution constraint is not "
        "supported yet"
    )
    assert expected in text
    missing = FamilyCatalog.unavailable("registry offline")
    assert missing.describe() == "family catalog unavailable: registry offline"


def test_load_catalog_success_then_cache() -> None:
    harness = _Harness(_Result(0, json.dumps(_document()), ""))
    cache: dict[str, FamilyCatalog] = {}
    first = load_catalog(harness, cache=cache)
    second = load_catalog(harness, cache=cache)
    assert first.source == "harness"
    assert first.available
    assert first.execution_constraint is False
    assert "not supported yet" in first.note
    assert second.source == "cached"
    assert second.required == first.required
    assert harness.calls == 1


def test_load_catalog_nonzero_exit_uses_stderr() -> None:
    catalog = load_catalog(_Harness(_Result(3, "", "catalog exploded: missing registry")))
    assert not catalog.available
    assert "catalog exploded: missing registry" in catalog.note
    assert catalog.describe() == ("family catalog unavailable: catalog exploded: missing registry")


def test_load_catalog_truncates_long_reasons() -> None:
    catalog = load_catalog(_Harness(_Result(1, "", "E" * 400)))
    assert catalog.note == "E" * 200


def test_load_catalog_garbage_stdout_is_unavailable() -> None:
    catalog = load_catalog(_Harness(_Result(0, "not-json", "")))
    assert not catalog.available
    assert catalog.note


def test_load_catalog_harness_exception_is_unavailable() -> None:
    catalog = load_catalog(_Harness(exc=RuntimeError("harness down")))
    assert not catalog.available
    assert "harness down" in catalog.note


def test_load_catalog_does_not_cache_failures() -> None:
    cache: dict[str, FamilyCatalog] = {}
    catalog = load_catalog(_Harness(_Result(1, "", "nope")), cache=cache)
    assert not catalog.available
    assert "catalog" not in cache


def test_matches_whole_words_and_underscore_spellings() -> None:
    catalog = _resolver_catalog()
    assert resolve_families("use conformal intervals", catalog).eligible[0].name == "conformal"
    assert not resolve_families("conformally smooth", catalog).mentioned
    assert not resolve_families("preconformal", catalog).mentioned
    for phrase in ("jackknife plus", "jackknife-plus", "jackknife_plus"):
        found = resolve_families(phrase, catalog)
        assert [mention.name for mention in found.eligible] == ["jackknife_plus"]


def test_aliases_resolve_to_catalog_families() -> None:
    catalog = _resolver_catalog()
    assert resolve_families("tail risk", catalog).eligible[0].name == "tail"
    assert resolve_families("CRPS", catalog).eligible[0].name == "distribution"
    assert resolve_families("order book", catalog).eligible[0].name == "northset"


@pytest.mark.parametrize(("phrase", "family"), sorted(DEFAULT_ALIASES.items()))
def test_each_default_alias(phrase: str, family: str) -> None:
    found = resolve_families(phrase, _resolver_catalog())
    assert [mention.name for mention in found.eligible] == [family]


def test_explicit_family_tokens_classify_eligible_retired_and_unknown() -> None:
    catalog = _resolver_catalog()
    eligible = resolve_families("please run family:conformal", catalog)
    assert eligible.eligible[0].name == "conformal"
    assert eligible.eligible[0].state is FamilyState.REQUIRED
    assert eligible.eligible[0].matched_text == "family:conformal"
    retired = resolve_families("family:old_family", catalog)
    assert retired.retired[0].state is FamilyState.RETIRED
    assert retired.explain() == [
        "family 'old_family' — retired; new selection blocked (historical receipts still verify)"
    ]
    unknown = resolve_families("family:foo_bar", catalog)
    assert unknown.unknown[0].state is FamilyState.UNKNOWN
    assert unknown.explain() == ["'foo_bar' — not a known family; no suitable family"]
    free = resolve_families("conformal and foo_bar", catalog)
    assert free.unknown == ()
    assert [mention.name for mention in free.eligible] == ["conformal"]


def test_refused_only_when_every_mention_is_blocked() -> None:
    catalog = _resolver_catalog()
    assert resolve_families("family:old_family", catalog).refused
    assert resolve_families("family:foo_bar", catalog).refused
    assert not resolve_families("conformal", catalog).refused
    mixed = resolve_families("conformal family:old_family", catalog)
    assert not mixed.refused
    assert mixed.eligible and mixed.retired
    silent = resolve_families("bananas", catalog)
    assert not silent.mentioned
    assert not silent.refused


def test_resolve_orders_by_first_appearance_and_dedupes() -> None:
    catalog = _resolver_catalog()
    found = resolve_families("CRPS, then tail risk, then conformal, then pinball", catalog)
    assert [mention.name for mention in found.eligible] == ["distribution", "tail", "conformal"]
    assert found.eligible[0].matched_text == "crps"


def test_longest_phrase_wins() -> None:
    catalog = _resolver_catalog()
    found = resolve_families("weighted conformal", catalog)
    assert [mention.name for mention in found.eligible] == ["weighted_conformal"]
    assert found.retired == ()
    assert found.unknown == ()


def test_optional_family_explain_line() -> None:
    found = resolve_families("candle order book", _resolver_catalog())
    assert found.explain() == ["family 'candle_order_book' — eligible (optional)"]


def test_unavailable_catalog_routes_mentions_to_unknown() -> None:
    catalog = FamilyCatalog.unavailable("registry offline")
    found = resolve_families("tail risk and family:conformal and bananas", catalog)
    assert found.catalog_source == "unavailable"
    assert found.eligible == ()
    assert found.retired == ()
    assert [mention.name for mention in found.unknown] == ["tail", "conformal"]
    assert all(mention.state is FamilyState.UNKNOWN for mention in found.unknown)
    assert found.explain() == [_CLOSED]
    assert not resolve_families("conformal", catalog).mentioned


def test_resolve_families_stays_fast_on_a_live_sized_catalog() -> None:
    names = [f"family_{i:04d}" for i in range(2700)]
    catalog = FamilyCatalog(
        schema_version=1,
        catalog_version=2,
        receipt_schema_version=2,
        required=frozenset(names[:23]),
        live_optional=frozenset(names[23:]),
        retired_count=7529,
        retired_sample=frozenset({"retired_sample_family"}),
        order=tuple(names[:23]),
        generated_at="2026-10-12T00:00:00Z",
    )
    goals = [
        f"score family_{i:04d} and family {(i + 9) % 2700:04d} family:missing_{i}"
        for i in range(50)
    ]
    started = time.perf_counter()
    selections = [resolve_families(goal, catalog) for goal in goals]
    elapsed = time.perf_counter() - started
    # Bare path stays under a second. Coverage tracing (the proofcore
    # fx1-coverage job) measured 3.9s for this same loop; a quadratic
    # resolver is still far above either cap.
    budget = 8.0 if "coverage" in sys.modules else 1.0
    assert elapsed < budget
    first = selections[0]
    assert [mention.name for mention in first.eligible] == ["family_0000", "family_0009"]
    assert first.unknown[0].name == "missing_0"


def test_families_prompt_block_empty_and_populated() -> None:
    catalog = _resolver_catalog()
    assert families_prompt_block(resolve_families("bananas", catalog), catalog) == ""
    block = families_prompt_block(resolve_families("tail risk then conformal", catalog), catalog)
    lines = block.splitlines()
    assert lines[0] == "RESEARCH FAMILIES (catalog v2, source harness):"
    assert lines[1] == "family 'tail' — eligible (required)"
    assert lines[2] == "family 'conformal' — eligible (required)"
    assert lines[-1] == _SCORECARD_NOTE
    locked = replace(catalog, execution_constraint=True)
    locked_block = families_prompt_block(resolve_families("conformal", catalog), locked)
    assert "family 'conformal' — eligible (required)" in locked_block
    assert "not the benches executed" not in locked_block


def test_missing_retired_list_keeps_sample_only_classification() -> None:
    catalog = FamilyCatalog.from_json(json.dumps(_document()))
    assert catalog.retired == frozenset()
    assert catalog.state_of("old_family") is FamilyState.RETIRED
    assert catalog.state_of("not_in_sample_family") is FamilyState.UNKNOWN
    found = resolve_families("family:not_in_sample_family", catalog)
    assert found.unknown[0].state is FamilyState.UNKNOWN


def test_full_retired_list_classifies_names_outside_the_sample() -> None:
    outside = "zz_not_in_sample_family"
    retired = sorted([*(f"retired_{index:04d}" for index in range(4000)), outside])
    assert outside not in retired[:20]
    document = _document()
    document["retired"] = retired
    document["retired_count"] = len(retired)
    document["retired_sample"] = retired[:20]
    catalog = FamilyCatalog.from_json(json.dumps(document))
    assert catalog.state_of(outside) is FamilyState.RETIRED
    started = time.perf_counter()
    found = resolve_families(f"please score {outside} today", catalog)
    elapsed = time.perf_counter() - started
    budget = 1.0 if "coverage" in sys.modules else 0.05
    assert elapsed < budget
    assert [mention.name for mention in found.retired] == [outside]
    explicit = resolve_families(f"family:{outside}", catalog)
    assert explicit.retired[0].state is FamilyState.RETIRED
    block = families_prompt_block(found, catalog)
    assert outside in block
    assert "retired_0000" not in block
    assert "retired_3999" not in block


def test_retired_list_must_match_count_sample_and_order() -> None:
    document = _document()
    document["retired"] = ["dead_bench", "old_family"]
    document["retired_count"] = 1
    with pytest.raises(ValueError, match="retired_count"):
        FamilyCatalog.from_json(json.dumps(document))
    document["retired_count"] = 2
    document["retired_sample"] = ["old_family", "dead_bench"]
    with pytest.raises(ValueError, match="retired_sample"):
        FamilyCatalog.from_json(json.dumps(document))
    document["retired"] = ["old_family", "dead_bench"]
    document["retired_sample"] = ["old_family", "dead_bench"]
    with pytest.raises(ValueError, match="sorted"):
        FamilyCatalog.from_json(json.dumps(document))
    document["retired"] = ["dead_bench", "dead_bench"]
    document["retired_sample"] = ["dead_bench", "dead_bench"]
    with pytest.raises(ValueError, match="duplicates"):
        FamilyCatalog.from_json(json.dumps(document))
