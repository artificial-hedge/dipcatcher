"""Pins for ``_harvest_p_values``, re-exported from ``research_cmds``.

The online-FDR command feeds this helper every harvested finding. These cases
lock the selection rules (mapping vs attribute, bool exclusion, order).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

from quant_fund.cli.research_cmds import _harvest_p_values


class _Bag(Mapping[str, object]):
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> object:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_empty_and_non_p_findings_contribute_nothing() -> None:
    assert _harvest_p_values([]) == []
    assert _harvest_p_values(()) == []
    assert (
        _harvest_p_values(
            [
                {"stat": "e", "value": 0.01},
                {"stat": "P", "value": 0.01},
                {"stat": "p"},
                {"value": 0.01},
                {"stat": "p", "value": "0.01"},
                {"stat": "p", "value": None},
                "p",
                None,
                SimpleNamespace(stat="t", value=0.2),
                SimpleNamespace(value=0.2),
                SimpleNamespace(stat="p"),
            ]
        )
        == []
    )


def test_numeric_p_values_keep_order_and_become_floats() -> None:
    harvested = _harvest_p_values(
        [
            {"stat": "p", "value": 0},
            {"stat": "ic", "value": 0.9},
            {"stat": "p", "value": 1},
            {"stat": "p", "value": -0.5},
            {"stat": "p", "value": 2},
        ]
    )
    assert harvested == [0.0, 1.0, -0.5, 2.0]
    assert all(type(value) is float for value in harvested)


def test_bool_is_not_a_p_value() -> None:
    assert _harvest_p_values(
        [
            {"stat": "p", "value": True},
            {"stat": "p", "value": False},
            SimpleNamespace(stat="p", value=True),
            {"stat": "p", "value": 0},
        ]
    ) == [0.0]


def test_non_finite_floats_are_kept() -> None:
    harvested = _harvest_p_values(
        [{"stat": "p", "value": math.nan}, {"stat": "p", "value": math.inf}]
    )
    assert len(harvested) == 2
    assert math.isnan(harvested[0])
    assert harvested[1] == math.inf


def test_research_cmds_reexports_the_same_function() -> None:
    from quant_fund.cli import research_pvalues
    from quant_fund.cli.research_cmds import _harvest_p_values as exported

    assert exported is research_pvalues._harvest_p_values


def test_mapping_wins_over_attributes_and_generators_work() -> None:
    class _Both(dict[str, object]):
        stat = "nope"
        value = 1

    bag = _Bag({"stat": "p", "value": 0.25})
    both = _Both({"stat": "p", "value": 0.5})
    found = _harvest_p_values(
        item
        for item in (
            bag,
            both,
            SimpleNamespace(stat="p", value=3),
            {"stat": "e", "value": 0.1},
        )
    )
    assert found == [0.25, 0.5, 3.0]
