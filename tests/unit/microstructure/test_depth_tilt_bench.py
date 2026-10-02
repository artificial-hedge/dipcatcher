"""depth_tilt bench — tilt-path structure + receipt seal."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.depth_tilt_bench import (
    depth_tilt_bench,
    lobster_tilt_path,
    sim_tilt_path,
)


def _write_tape(tmp_path: Path, n: int = 40) -> tuple[Path, Path]:
    """Synthetic LOBSTER pair: each even row is a buy-initiated exec; after
    each exec the bid touch thickens (accommodation)."""
    msg = tmp_path / "AMZN_2012-06-21_1_2_message_1.csv"
    ob = tmp_path / "AMZN_2012-06-21_1_2_orderbook_1.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        for i in range(n):
            is_exec = i % 2 == 0 and i > 0
            etype = 4 if is_exec else 1
            wm.writerow([36000 + i, etype, 1, 10, 40100, -1])
            if is_exec:
                wo.writerow([40100, 100, 40000, 100])
            else:
                # post-fill accommodation: the row after each exec is bid-heavy
                wo.writerow([40100, 50, 40000, 400])
    return msg, ob


def test_lobster_tilt_path_accommodation(tmp_path: Path) -> None:
    msg, ob = _write_tape(tmp_path)
    r = lobster_tilt_path(msg, ob, horizons=(1, 2))
    assert r["ok"]
    assert r["n_fills"] == 19
    # bid-heavy rows (50 ask vs 400 bid -> tilt +0.777...) signed +1
    assert r["tilt_path"]["1"] == pytest.approx(0.7777, abs=1e-3)
    assert r["tilt_base"] is not None


def test_sim_tilt_path_structure() -> None:
    r = sim_tilt_path(horizon=800, horizons=(1, 5))
    assert r["ok"]
    assert set(r["tilt_path"]) == {"1", "5"}
    assert r["tilt_base"] is not None


def test_bench_seals_and_verifies(tmp_path: Path) -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_file

    msg, ob = _write_tape(tmp_path, n=60)
    payload = depth_tilt_bench(tmp_path, "AMZN", horizon=400, seed=3)
    assert payload["schema"] == "depth_tilt.v1"
    assert set(payload["sim_arms"]) == {"iid", "split", "lv_cd300", "lv_cd300_narrow"}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(__import__("json").dumps(payload))
    result = verify_receipt_file(receipt)
    assert result.get("valid") is True


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        depth_tilt_bench(tmp_path, "AMZN", horizon=60, seed=1)


def test_lo_tilt_validation_and_response() -> None:
    from dataclasses import replace

    from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config

    with pytest.raises(ValueError, match="lo_tilt_decay"):
        replace(santa_fe_config(), lo_tilt_decay=1.5)
    with pytest.raises(ValueError, match="lo_tilt_gain"):
        replace(santa_fe_config(), lo_tilt_gain=-0.1)

    sim = ZILobSimulator(replace(santa_fe_config(seed=5), lo_tilt_gain=0.5, lo_tilt_decay=0.0))
    for _ in range(3000):
        sim.step()
    assert sim._tilt != 0.0 or not sim.trades


def test_lo_tilt_zero_is_bit_identical() -> None:
    from dataclasses import replace

    from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config

    a = ZILobSimulator(santa_fe_config(seed=13))
    b = ZILobSimulator(replace(santa_fe_config(seed=13), lo_tilt_gain=0.0))
    for _ in range(1500):
        a.step()
        b.step()
    assert [(t.price, t.level, t.aggressor) for t in a.trades] == [
        (t.price, t.level, t.aggressor) for t in b.trades
    ]


def test_hit_narrow_validation_and_marker() -> None:
    from dataclasses import replace

    from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config

    for name, bad in (("hit_narrow_dist", -1), ("hit_narrow_window", -2)):
        with pytest.raises(ValueError, match=name):
            replace(santa_fe_config(), **{name: bad})

    sim = ZILobSimulator(replace(santa_fe_config(seed=7), hit_narrow_dist=2, hit_narrow_window=40))
    for _ in range(2000):
        sim.step()
    if sim.trades:
        assert sim._hit_retreat is None or sim._hit_retreat[1] > sim.n_events - 1
