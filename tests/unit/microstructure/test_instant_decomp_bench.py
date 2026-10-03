"""Tests for instant_decomp_bench — touch-emptying attribution."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.instant_decomp_bench import (
    INSTANT_DECOMP_SCHEMA,
    _decomp_stats,
    instant_decomp_bench,
    lobster_instant_decomp,
    sim_instant_decomp,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def test_decomp_stats_identity() -> None:
    # 10 fills, 5 emptied with mean gap 3.0 ticks; realized instant 1.5.
    out = _decomp_stats(
        n_fills=10,
        n_empty=5,
        gap_ticks=[2.0, 4.0, 2.0, 4.0, 3.0],
        signed_dmids=[1.5] * 10,
    )
    assert out["ok"] is True
    assert out["p_empty_touch"] == pytest.approx(0.5)
    assert out["mean_gap_ticks_when_empty"] == pytest.approx(3.0)
    assert out["predicted_instant_ticks"] == pytest.approx(0.75)
    assert out["realized_instant_ticks"] == pytest.approx(1.5)
    assert out["unexplained_instant_ticks"] == pytest.approx(0.75)


def test_decomp_stats_empty() -> None:
    assert _decomp_stats(n_fills=0, n_empty=0, gap_ticks=[], signed_dmids=[])["ok"] is False


def test_lobster_instant_decomp_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        # Seed row: ask 4000 x200, bid 3900 x100 (LOBSTER 1/10000$ units).
        wm.writerow([34200.0, 1, 1, 100, 2500, 1])
        wo.writerow([4000, 200, 3900, 100])
        oid = 10
        for i in range(12):
            # Buyer-initiated exec (dir -1) against the ask at 4000.
            wm.writerow([34201.0 + i, 4, 100 + i, 200, 4000, -1])
            if i % 2 == 0:
                # Emptied: next ask sits 1 tick higher (4100).
                wo.writerow([4100, 150, 3900, 100])
                # Restore submission: a new ask order re-posts 4000 so
                # the next exec's pre-touch is 4000 again.
                oid += 1
                wm.writerow([34201.2 + i, 1, 2000 + oid, 150, 4000, -1])
                wo.writerow([4000, 150, 3900, 100])
            else:
                # Not emptied: the touch stays 4000.
                wo.writerow([4000, 150, 3900, 100])
    out = lobster_instant_decomp(msg, ob)
    assert out["ok"] is True
    assert out["n_fills"] == 12
    assert out["p_empty_touch"] == pytest.approx(0.5)
    assert out["mean_gap_ticks_when_empty"] == pytest.approx(1.0)
    # Emptied fills: ask 4000->4100 => mid +0.5 ticks. Non-emptied: flat.
    assert out["realized_instant_ticks"] == pytest.approx(0.25)
    assert out["predicted_instant_ticks"] == pytest.approx(0.25)
    assert out["unexplained_instant_ticks"] == pytest.approx(0.0)


def test_sim_instant_decomp_runs() -> None:
    out = sim_instant_decomp(horizon=3000, seed=3)
    assert out["ok"] is True
    assert out["n_fills"] > 0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        instant_decomp_bench(tmp_path)


def test_bench_seals(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    rows = [
        ([34200.0 + i * 0.5, 1, i + 1, 100, 2500, 1], [4000 - i, 200, 3900, 100]) for i in range(15)
    ]
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        for m, o in rows:
            wm.writerow(m)
            wo.writerow(o)
    out = instant_decomp_bench(tmp_path, horizon=800, seed=3)
    assert out["schema"] == INSTANT_DECOMP_SCHEMA
    assert out["research_only"] is True
    seal = out.pop("receipt_sha256")
    assert hash_bytes(canonical_json_bytes(out)) == seal
