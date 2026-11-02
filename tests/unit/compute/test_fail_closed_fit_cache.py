"""FitCache refuses a capacity that is not a positive integer."""

from __future__ import annotations

import pytest

from quant_fund.compute.cache import FitCache


def test_positive_capacity_still_stores() -> None:
    cache = FitCache(max_entries=1)
    cache.put("k", {"snapshot": 1})
    assert cache.get("k") == {"snapshot": 1}
    cache.put("other", {"snapshot": 2})
    assert cache.get("k") is None


@pytest.mark.parametrize("capacity", [0, -1, True, False, 1.5, float("nan"), float("inf")])
def test_capacity_must_be_a_positive_integer(capacity: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FitCache(max_entries=capacity)  # type: ignore[arg-type]
