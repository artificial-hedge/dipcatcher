"""Tests for microstructure/markout.py — post-execution markout lane.

Signed markout = aggressor-direction mid move in ticks over {0.05..30}s on
the LOBSTER tape and on three SYNTHETIC ZI-LOB flow arms. Crafted-tape tests
pin the exact curve arithmetic (searchsorted horizon lookup, resting-side
sign flip, horizon overrun dropping n); the bench receipt must verify clean
and carry no forbidden headline-metric keys.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.markout import (
    HORIZONS_S,
    MARKOUT_CLAIM,
    MARKOUT_SCHEMA,
    SplitFlow,
    markout_bench,
    markout_curve,
    sim_markout_curve,
    tape_markout_curve,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent
from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

# (t_seconds_after_midnight, mid_ticks) rows of the crafted tape. mid_ticks is
# the top-of-book mid in cent units; ob rows are emitted with a 100-tick spread.
_CRAFT: list[tuple[float, float]] = [
    (34200.5, 1000.0),
    (34200.6, 1000.0),
    (34201.0, 1000.0),  # exec A: dir +1 (resting buy hit) -> SELL aggressor
    (34201.05, 1001.0),
    (34201.1, 1002.0),
    (34201.25, 1003.0),
    (34201.5, 1004.0),
    (34202.0, 1005.0),  # exec B: dir -1 (resting sell hit) -> BUY aggressor
    (34202.05, 1006.0),
    (34202.1, 1007.0),
    (34202.25, 1008.0),
    (34202.5, 1009.0),
    (34203.0, 1010.0),
    (34204.0, 1011.0),
    (34207.0, 1012.0),
    (34212.0, 1013.0),
    (34232.0, 1014.0),  # exec C: dir -1 -> BUY aggressor, short horizons only
    (34233.0, 1015.0),
]
_CRAFT_EXECS = {2: 1, 7: -1, 16: -1}  # row index -> LOBSTER direction

# Exact signed markouts (ticks) read off the crafted grid:
#   exec A (sell) anchors mid 1000 @34201.0 -> -1,-2,-3,-4,-5,-10,-12,-13,-14
#   exec B (buy)  anchors mid 1005 @34202.0 -> +1,+2,+3,+4,+5,+6,+7,+8,+9
#   exec C (buy)  anchors mid 1014 @34232.0 -> +1 for h<=1, then overrun
_BUY_MEANS = [1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 7.0, 8.0, 9.0]
_SELL_MEANS = [-1.0, -2.0, -3.0, -4.0, -5.0, -10.0, -12.0, -13.0, -14.0]
_POOLED_N = [3, 3, 3, 3, 3, 2, 2, 2, 2]


def _write_tape(
    tmp_path: Path,
    rows: list[tuple[float, float]],
    exec_dirs: dict[int, int],
) -> tuple[Path, Path]:
    """Emit a minimal LOBSTER-format (message, orderbook_1-level) CSV pair."""
    msg_path = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_1.csv"
    ob_path = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
    with msg_path.open("w", newline="") as f_msg, ob_path.open("w", newline="") as f_ob:
        msg_w = csv.writer(f_msg)
        ob_w = csv.writer(f_ob)
        oid = 100
        for i, (t, mid_ticks) in enumerate(rows):
            raw_mid = int(round(mid_ticks * 100))
            if i in exec_dirs:
                direction = exec_dirs[i]
                msg_w.writerow([f"{t}", "4", str(oid), "50", str(raw_mid), str(direction)])
            else:
                side = 1 if i % 2 else -1
                msg_w.writerow([f"{t}", "1", str(oid), "10", str(raw_mid), str(side)])
            oid += 1
            ob_w.writerow([str(raw_mid + 50), "10", str(raw_mid - 50), "10"])
    return msg_path, ob_path


def test_crafted_tape_exact_curve(tmp_path: Path) -> None:
    msg, ob = _write_tape(tmp_path, _CRAFT, _CRAFT_EXECS)
    curve = tape_markout_curve(msg, ob)
    assert curve["horizons_s"] == list(HORIZONS_S)
    assert curve["n_executions"] == 3
    assert curve["n_book_events"] == len(_CRAFT)

    buy = curve["sides"]["buy"]
    sell = curve["sides"]["sell"]
    assert buy["mean"] == pytest.approx(_BUY_MEANS)
    assert sell["mean"] == pytest.approx(_SELL_MEANS)
    # exec C overruns the tape for h>1s: n drops from 2 to 1 on the buy side.
    assert buy["n"] == [2, 2, 2, 2, 2, 1, 1, 1, 1]
    assert sell["n"] == [1] * 9
    assert curve["signed"]["n"] == _POOLED_N
    assert curve["signed"]["median"][0] == pytest.approx(1.0)  # median of -1,+1,+1
    # Pooled mean never clears its SEM floor on this symmetric fixture.
    assert curve["toxicity_half_life_s"] is None


def test_crafted_tape_half_life_onset(tmp_path: Path) -> None:
    # All buy-aggressor executions on a rising mid: pooled mean is uniformly
    # positive, so the first horizon already clears the noise floor.
    execs = {2: -1, 7: -1, 16: -1}
    msg, ob = _write_tape(tmp_path, _CRAFT, execs)
    curve = tape_markout_curve(msg, ob)
    assert curve["toxicity_half_life_s"] == pytest.approx(0.05)
    assert curve["sides"]["sell"]["n"] == [0] * 9


def test_markout_curve_input_validation() -> None:
    with pytest.raises(ValueError, match="empty sample series"):
        markout_curve(
            np.asarray([]), np.asarray([]), np.asarray([], dtype=np.int64), np.asarray([])
        )
    with pytest.raises(ValueError, match="equal-length"):
        markout_curve(
            np.asarray([0.0, 1.0]),
            np.asarray([10.0, 11.0]),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        markout_curve(
            np.asarray([0.0, 1.0]),
            np.asarray([10.0, 11.0]),
            np.asarray([0], dtype=np.int64),
            np.asarray([1.0]),
            horizons=(-1.0,),
        )


def test_missing_tape_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        markout_bench(tmp_path, sim_seconds=30.0)
    (tmp_path / "AMZN_2012-06-21_34200000_57600000_message_1.csv").write_text(
        "34200.0,1,1,10,100000,1\n"
    )
    with pytest.raises(FileNotFoundError):
        markout_bench(tmp_path, sim_seconds=30.0)


def test_sim_arm_deterministic() -> None:
    curve_a = sim_markout_curve(ZILobSimulator(ZILobConfig(seed=11)), sim_seconds=300.0)
    curve_b = sim_markout_curve(ZILobSimulator(ZILobConfig(seed=11)), sim_seconds=300.0)
    assert curve_a["n_executions"] > 10
    assert json.dumps(curve_a, sort_keys=True) == json.dumps(curve_b, sort_keys=True)
    assert curve_a["sides"]["buy"]["n"][0] + curve_a["sides"]["sell"]["n"][0] == int(
        curve_a["signed"]["n"][0]
    )


def test_split_flow_episodes() -> None:
    quiet = SplitFlow(p_start=0.0, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0)
    for _ in range(200):
        assert not quiet.in_episode
        assert quiet.current().name == "idle"
        quiet.advance()
    assert quiet.n_episodes == 0

    bursty = SplitFlow(p_start=1.0, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0)
    bursty.advance()
    assert bursty.in_episode
    state = bursty.current()
    assert state.intensity_mult == 3.0
    assert state.p_buy in (0.0, 1.0)
    k_seen = bursty.episode_lengths[0]
    assert 10 <= k_seen <= 600

    again = SplitFlow(p_start=0.5, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0)
    again2 = SplitFlow(p_start=0.5, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0)
    for _ in range(500):
        again.advance()
        again2.advance()
    assert again.episode_lengths == again2.episode_lengths


def test_divergence_gate() -> None:
    from quant_fund.microstructure.markout import _divergences

    real = {"signed": {"mean": [None] * 6 + [10.0, None, None]}}
    arms = {
        "close": {"signed": {"mean": [None] * 6 + [7.0, None, None]}},  # gap 3 < 5
        "far": {"signed": {"mean": [None] * 6 + [16.0, None, None]}},  # gap 6 > 5
        "empty": {"signed": {"mean": [None] * 9}},
    }
    out = {d["arm"]: d for d in _divergences(real, arms, HORIZONS_S)}
    assert out["close"]["diverges"] is False
    assert out["close"]["abs_gap_ticks"] == pytest.approx(3.0)
    assert out["far"]["diverges"] is True
    assert out["empty"]["diverges"] is False


def test_bench_receipt_seals_and_verifies(tmp_path: Path) -> None:
    _write_tape(tmp_path, _CRAFT, _CRAFT_EXECS)
    payload = markout_bench(tmp_path, sim_seconds=120.0)

    assert payload["schema"] == MARKOUT_SCHEMA
    assert payload["kind"] == "markout"
    assert payload["claim"] == MARKOUT_CLAIM
    assert payload["data_label"] == "MIXED"
    assert payload["research_only"] is True
    assert set(payload["sim_arms"]) == {"iid", "markov_regime", "split"}
    assert {d["arm"] for d in payload["divergences"]} == {"iid", "markov_regime", "split"}

    # Seal convention: sha256 of the canonical payload minus the seal field.
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    assert payload["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))
    assert family_blob_forbidden_metrics_absent(payload)
    # No NaN/Infinity may survive into the sealed payload.
    json.dumps(payload, allow_nan=False)

    result = verify_receipt_payload(payload, tmp_path / "markout_test.json")
    assert result["valid"], result["errors"]
    assert result["errors"] == []
