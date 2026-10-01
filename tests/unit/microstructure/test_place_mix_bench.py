"""place_mix_bench tests — synthetic-tape mixture fit + mechanism pins."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.place_law_bench import _calibrated, _DistSim
from quant_fund.microstructure.place_mix_bench import (
    PLACE_MIX_SCHEMA,
    place_mix_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_file


def _write_tape(tmp_path: Path, ticker: str = "TST") -> Path:
    """Minimal LOBSTER pair: seeded book + a run of LO submissions."""
    n_msg = 400
    msg_path = tmp_path / f"{ticker}_2020-01-02_09300000_16000000_message_10.csv"
    ob_path = tmp_path / f"{ticker}_2020-01-02_09300000_16000000_orderbook_10.csv"
    rng = np.random.default_rng(5)
    ask, bid = 10000, 9900  # level-10 units ($0.0001); 100 = one tick
    rows: list[list[str]] = []
    ob_rows: list[list[str]] = []
    next_id = 1
    # Seed: 10 resting bids/asks so the book is non-empty.
    for k in range(10):
        rows.append([f"{0.001 * (k + 1):.6f}", "1", str(next_id), "50", str(bid - k * 100), "1"])
        next_id += 1
        rows.append(
            [f"{0.001 * (k + 1) + 0.0005:.6f}", "1", str(next_id), "50", str(ask + k * 100), "-1"]
        )
        next_id += 1
    for j in range(n_msg - 20):
        u = float(rng.random())
        side = rng.choice(["1", "-1"])
        if u < 0.12:
            # join at the own-side touch (d = 0)
            price = bid if side == "1" else ask
        elif u < 0.22:
            # inside the spread (d < 0): between bid and ask
            price = (
                int(rng.integers(bid + 10, ask))
                if side == "1"
                else int(rng.integers(bid + 1, ask - 10))
            )
        else:
            d = int(rng.integers(1, 16))
            price = bid - d * 100 if side == "1" else ask + d * 100
        rows.append([f"{0.05 + j * 0.001:.6f}", "1", str(next_id), "25", str(price), side])
        next_id += 1
    # Emit orderbook rows (10 levels, interleaved ap, asz, bp, bsz).
    for _j in range(len(rows)):
        row: list[str] = []
        for k in range(10):
            row += [str(ask + k * 100), "50", str(bid - k * 100), "50"]
        ob_rows.append(row)
    with msg_path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with ob_path.open("w", newline="") as fh:
        csv.writer(fh).writerows(ob_rows)
    return tmp_path


def test_join_frac_places_at_touch() -> None:
    """place_join_frac=1.0 under ref anchor puts every LO at the touch."""
    sim = _DistSim(_calibrated(0, {"place_join_frac": 1.0}), None)
    for _ in range(1500):
        sim.step()
    assert sim.dist_log
    assert set(sim.dist_log) == {0.0}
    assert sim.n_lo_join == len(sim.dist_log)


def test_mixture_knobs_bit_identical_at_zero() -> None:
    """place_join_frac/lo_improve_frac at 0 leave the ref path identical."""
    a = _DistSim(_calibrated(0, {}), None)
    b = _DistSim(_calibrated(0, {"place_join_frac": 0.0, "lo_improve_frac": 0.0}), None)
    for _ in range(1200):
        a.step()
        b.step()
    assert a.dist_log == b.dist_log
    assert a.n_events == b.n_events


def test_improve_under_ref_anchor() -> None:
    """lo_improve_frac>0 under ref anchor emits strictly-inside levels."""
    sim = _DistSim(_calibrated(0, {"lo_improve_frac": 0.9}), None)
    for _ in range(3000):
        sim.step()
    assert sim.n_lo_improve > 0
    assert any(d < 0 for d in sim.dist_log)


def test_join_frac_validated() -> None:
    with pytest.raises(ValueError, match="place_join_frac"):
        _calibrated(0, {"place_join_frac": 1.5})


def test_place_mix_bench_seals(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    receipt = place_mix_bench(tape, "TST", horizon=600, scan_horizon=400)
    assert receipt["schema"] == PLACE_MIX_SCHEMA
    assert receipt["claims"]["tape_has_three_components"]
    out = tmp_path / "place_mix_tst.json"
    out.write_text(__import__("json").dumps(receipt))
    result = verify_receipt_file(out)
    assert result["valid"], result


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        place_mix_bench(tmp_path, "NOPE", horizon=100, scan_horizon=50)
