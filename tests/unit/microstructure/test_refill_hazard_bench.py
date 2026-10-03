"""Tests for refill_hazard_bench: level-memory hazard on tape + sim."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.microstructure.refill_hazard_bench import (
    REFILL_HAZARD_SCHEMA,
    lobster_refill_hazard,
    refill_hazard_bench,
    sim_refill_hazard,
)


def _write_pair(
    tmp_path: Path,
    messages: list[list[Any]],
    book_rows: list[list[int]],
) -> tuple[Path, Path]:
    msg = tmp_path / "AMZN_x_message_y.csv"
    ob = tmp_path / "AMZN_x_orderbook_y.csv"
    with msg.open("w", newline="") as f:
        csv.writer(f).writerows(messages)
    with ob.open("w", newline="") as f:
        csv.writer(f).writerows(book_rows)
    return msg, ob


def test_lobster_refill_delay_and_touch_return(tmp_path: Path) -> None:
    """12 touch-emptying execs: even-indexed ones refill one row later."""
    # Row layout is interleaved 4 cols per level: [ask_p, ask_sz, bid_p, bid_sz].
    messages: list[list[Any]] = []
    book_rows: list[list[int]] = []
    oid = 10
    t = 34200.0
    for i in range(12):
        p = 40000 + i * 100  # ask touch price emptied this round
        # pre row: two ask levels [p, p+100] -> pre_gap = 1 tick
        oid += 1
        messages.append([t, 1, oid, 100, p, 1])
        book_rows.append([p, 100, 3900, 200, p + 100, 30, 0, 0])
        # exec empties the ask touch p
        oid += 1
        messages.append([t + 0.1, 4, oid, 100, p, -1])
        book_rows.append([p + 100, 30, 3900, 200])
        if i % 2 == 0:
            # an improving ask submission re-occupies p -> new touch
            oid += 1
            messages.append([t + 0.2, 1, oid, 25, p, 1])
            book_rows.append([p, 25, 3900, 200, p + 100, 30, 0, 0])
        t += 1.0
    msg, ob = _write_pair(tmp_path, messages, book_rows)
    stats = lobster_refill_hazard(msg, ob, max_delay=5)
    assert stats["ok"] is True
    assert stats["n_empty"] == 12
    assert stats["hazard_cdf"]["1"] == pytest.approx(0.5)
    assert stats["p_never_within_cap"] == pytest.approx(0.5)
    assert stats["p_refill_as_touch"] == pytest.approx(1.0)
    assert stats["median_delay_events"] == pytest.approx(1.0)
    assert set(stats["by_pre_gap"]) == {"gap_1"}


def test_sim_arm_hazard_structure() -> None:
    stats = sim_refill_hazard(horizon=3000, seed=3, max_delay=100)
    assert stats["ok"] is True
    assert stats["n_empty"] > 10
    assert "hazard_cdf" in stats
    assert stats["median_delay_events"] is not None


def test_bench_receipt_seals_and_verifies(tmp_path: Path) -> None:
    messages = [[34200.0, 1, 11, 100, 400000, 1]]
    book_rows = [[4000, 100, 3900, 200]]
    _write_pair(tmp_path, messages, book_rows)
    payload = refill_hazard_bench(tmp_path, "AMZN", horizon=60, seed=3, max_delay=10)
    assert payload["schema"] == REFILL_HAZARD_SCHEMA
    assert payload["kind"] == "refill_hazard_bench"
    assert payload["data_label"] == "MIXED"
    assert payload["research_only"] is True
    assert "receipt_sha256" in payload
    from quant_fund.research.receipt_v2 import verify_receipt_file

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(payload))
    result = verify_receipt_file(receipt_path)
    assert result.get("valid") is True


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        refill_hazard_bench(tmp_path, "AMZN", horizon=60, seed=1)


def test_refill_cooldown_validation() -> None:
    from dataclasses import replace

    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    with pytest.raises(ValueError, match="refill_cooldown"):
        replace(santa_fe_config(), refill_cooldown=-1)


def test_refill_cooldown_suppresses_vacated_level() -> None:
    """A level emptied by a fill rejects ZI placements inside the window."""
    from dataclasses import replace

    from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config

    cfg = replace(santa_fe_config(seed=5), anchor="ref", band=8, refill_cooldown=50)
    sim = ZILobSimulator(cfg)
    for _ in range(2000):
        sim.step()
    assert sim._vacancy, "no level ever emptied under this seed"
    # every vacated level within the window must be unoccupied or re-seeded
    # only by paths other than ZI flow (none exist here) -> all cooled
    # levels remain empty
    cooled = [(s, lv) for (s, lv), t0 in sim._vacancy.items() if sim.n_events - t0 < 50]
    book = {"buy": sim._bids, "sell": sim._asks}
    for side, lv in cooled:
        assert lv not in book[side]


def test_refill_cooldown_zero_is_bit_identical() -> None:
    """cd=0 consumes no extra draws: identical fill stream as legacy."""
    from dataclasses import replace

    from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config

    a = ZILobSimulator(santa_fe_config(seed=11))
    b = ZILobSimulator(replace(santa_fe_config(seed=11), refill_cooldown=0))
    for _ in range(1500):
        a.step()
        b.step()
    assert [(t.price, t.level, t.aggressor) for t in a.trades] == [
        (t.price, t.level, t.aggressor) for t in b.trades
    ]
    assert b.n_lo_suppressed == 0
