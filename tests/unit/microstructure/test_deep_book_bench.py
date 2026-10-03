"""deep_book_bench contracts."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.deep_book_bench import deep_book_bench


@pytest.mark.slow
def test_bench_seals_and_deep_beats_thin() -> None:
    p = deep_book_bench(horizon=200.0, seed=3)
    assert p["schema"] == "deep_book_bench.v1"
    assert p["data_label"] == "MIXED"
    assert p["research_only"] is True
    assert p["claims"]["deep_book_holds_depth"]
    assert p["claims"]["spread_enters_real_band"]
    assert len(p["receipt_sha256"]) == 64


def test_arms_present() -> None:
    p = deep_book_bench(horizon=60.0, seed=5)
    assert set(p["arms"]) == {"thin", "deep"}
    for a in p["arms"].values():
        assert a["spread_mean"] is not None or a["n_samples"] == 0
