"""Contract tests for lo_response_bench — post-fill LO channel decomposition."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quant_fund.microstructure.lo_response_bench import (
    LO_RESPONSE_SCHEMA,
    lo_response_bench,
    lobster_lo_response,
)
from quant_fund.research.receipt_v2 import verify_receipt_file


def _write_tape(tmp_path: Path, n: int = 400) -> tuple[Path, Path]:
    """Synthetic LOBSTER pair: execs balanced buy/sell, subs tiltable."""
    msg = tmp_path / "X_2020-01-01_0_1_message_10.csv"
    ob = tmp_path / "X_2020-01-01_0_1_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        mw, ow = csv.writer(fm), csv.writer(fo)
        for i in range(n):
            if i % 10 == 4:
                # execution: direction -1 => resting sell side consumed (buy fill)
                mw.writerow([i + 1.0, 4, i, 25, 1050000, -1])
                bid, ask = 1040000, 1060000
            elif i % 10 == 9:
                mw.writerow([i + 1.0, 4, i, 25, 1040000, 1])
                bid, ask = 1040000, 1060000
            else:
                # submission: after buy fills tilt toward bids (unhit side)
                direction = 1 if (i % 3) else -1
                px = 1045000 if direction == 1 else 1055000
                mw.writerow([i + 1.0, 1, i, 40, px, direction])
                bid, ask = 1045000, 1055000
            ow.writerow(
                x
                for k in range(10)
                for x in (
                    f"{ask + k * 100}",
                    80 if k == 0 else 100,
                    f"{bid - k * 100}",
                    90 if k == 0 else 100,
                )
            )
    return msg, ob


def test_lobster_lo_response_shape(tmp_path: Path) -> None:
    msg, ob = _write_tape(tmp_path, n=200)
    r = lobster_lo_response(msg, ob, horizons=(1, 5, 20))
    assert r["ok"] is True
    assert r["n_execs"] > 0
    p = r["post_fill"]["20"]
    assert p["n_submissions"] > 0
    assert 0.0 <= (p["share_unhit"] or 0.0) <= 1.0
    assert p["mean_dist_unhit"] is not None
    assert r["unconditional"]["n_submissions"] > 0


def test_lo_response_bench_seals(tmp_path: Path) -> None:
    _write_tape(tmp_path)
    payload = lo_response_bench(tmp_path, "X", horizon=400, seed=3)
    assert payload["schema"] == LO_RESPONSE_SCHEMA
    assert set(payload["sim_arms"]) == {"iid", "lv_cd300", "lv_cd300_narrow", "lv_cd300_tilt"}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload))
    result = verify_receipt_file(receipt)
    assert result.get("valid") is True


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lo_response_bench(tmp_path, "AMZN")


def test_trace_sim_ignores_seed_book() -> None:
    from quant_fund.microstructure.lo_response_bench import _TraceSim
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    sim = _TraceSim(santa_fe_config(seed=5), None)
    assert sim.lo_log == []
    sim.step()
    # whatever happened, entries only cover live events
    assert all(ev >= 1 for ev, _, _ in sim.lo_log)
