"""level_gap_bench — near-touch vacancy profile + placement-law scan."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from quant_fund.microstructure.level_gap_bench import (
    LEVEL_GAP_SCHEMA,
    level_gap_bench,
    lobster_level_gaps,
    sim_level_gaps,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def test_lobster_level_gaps(tmp_path: Path) -> None:
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with ob.open("w", newline="") as fo:
        w = csv.writer(fo)
        # Interleaved [ask_p, ask_sz, bid_p, bid_sz] per level:
        # asks 4000/4200/4500 (gaps 2, 3), bids 3900/3800 (gap 1).
        # Level-3 bid slot is empty (size 0 -> dropped by the parser).
        for _ in range(20):
            w.writerow([4000, 100, 3900, 200, 4200, 50, 3800, 80, 4500, 30, 0, 0])
    out = lobster_level_gaps(ob)
    assert out["ok"] is True
    assert out["n_rows"] == 20
    assert out["g1_ask"]["mean"] == pytest.approx(2.0)
    assert out["g1_bid"]["mean"] == pytest.approx(1.0)
    assert out["g2"]["mean"] == pytest.approx(3.0)
    assert out["spread_mean"] == pytest.approx(1.0)


def test_sim_level_gaps_runs() -> None:
    out = sim_level_gaps(horizon=2000, seed=3)
    assert out["ok"] is True
    assert out["n_rows"] == 2000
    assert out["g1_ask"]["ok"] is True
    assert out["g1_ask"]["mean"] >= 1.0  # occupied levels are >= 1 tick apart


def test_gap_stats_contract() -> None:
    # A wide-band, steep-exponent placement law must produce sparser
    # near-touch levels than the default flat placement.
    narrow = sim_level_gaps(horizon=3000, seed=5)
    wide = sim_level_gaps(
        replace(santa_fe_config(seed=5), density_exponent=1.5, band=40),
        horizon=3000,
        seed=5,
    )
    assert wide["g1_ask"]["mean"] > narrow["g1_ask"]["mean"]


def test_bench_seals(tmp_path: Path) -> None:
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with ob.open("w", newline="") as fo:
        w = csv.writer(fo)
        for _ in range(30):
            w.writerow([4000, 100, 3900, 200, 4100, 50, 3800, 80])
    out = level_gap_bench(tmp_path, horizon=800, seed=3)
    assert out["schema"] == LEVEL_GAP_SCHEMA
    body = dict(out)
    sha = body.pop("receipt_sha256")
    assert hash_bytes(canonical_json_bytes(body)) == sha
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(str(p))["valid"] is True
