"""Tests for quote_floor_bench."""

import pytest

from quant_fund.microstructure.quote_floor_bench import quote_floor_bench
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def test_quote_floor_shape() -> None:
    out = quote_floor_bench(horizon=3000, seed=7)
    assert out["schema"] == "quote_floor.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 6
    assert [c["min_quote_dist"] for c in out["cells"]] == [0, 4, 8, 12, 4, 8]
    assert [c["flow_intensity"] for c in out["cells"]] == [None, None, None, None, 2.0, 2.0]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "floor_opens_spread",
        "spread_scales_with_floor",
        "floor_composes",
    }
    assert len(out["receipt_sha256"]) == 64


def test_min_quote_dist_validation() -> None:
    with pytest.raises(ValueError, match="min_quote_dist"):
        ZILobConfig(min_quote_dist=-1)
    with pytest.raises(ValueError, match="min_quote_dist"):
        ZILobConfig(min_quote_dist=True)
