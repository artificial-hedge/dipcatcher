"""hidden_depth_bench + iceberg_reload contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.hidden_depth_bench import hidden_depth_bench
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)


def _run(reload_p: float, horizon: float = 300.0) -> ZILobSimulator:
    sim = ZILobSimulator(replace(santa_fe_config(seed=6), iceberg_reload=reload_p))
    while sim.t < horizon:
        sim.step()
    return sim


class TestIcebergReload:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="iceberg_reload"):
            replace(santa_fe_config(seed=0), iceberg_reload=1.5)

    def test_zero_bit_identical(self) -> None:
        a = _run(0.0)
        b = _run(0.0)
        assert [(t.aggressor, t.price, t.qty) for t in a.trades] == [
            (t.aggressor, t.price, t.qty) for t in b.trades
        ]
        assert a.n_hidden_fills == 0 == b.n_hidden_fills

    def test_reload_marks_hidden_fills(self) -> None:
        sim = _run(0.8)
        assert sim.n_hidden_fills > 0
        assert sim.event_counts()["n_hidden_fills"] == sim.n_hidden_fills


@pytest.mark.slow
def test_bench_seals() -> None:
    p = hidden_depth_bench(seed=3)
    assert p["schema"] == "hidden_depth_bench.v1"
    assert p["data_label"] == "MIXED"
    assert p["research_only"] is True
    assert p["claims"]["no_hidden_without_reload"]
    assert p["claims"]["monotone_in_reload"]
    assert len(p["receipt_sha256"]) == 64
