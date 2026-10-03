from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quant_fund.microstructure.imbalance_predict import (
    _lobster_samples,
    _measure,
    imbalance_predict_bench,
    sim_imbalance_predict,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _write_tape(
    tmp_path: Path, events: list[list[object]], rows: list[list[object]]
) -> tuple[Path, Path]:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        for e in events:
            wm.writerow(e)
        for r in rows:
            wo.writerow(r)
    return msg, ob


def test_lobster_samples_pairs_prior_row(tmp_path: Path) -> None:
    events = [
        [34200.0, 1, 1, 100, 4000000, 1],  # submission buy — event 0, no prior row
        [34201.0, 4, 2, 10, 4000100, -1],  # exec, resting sell → buy aggressor +1
        [34202.0, 4, 3, 10, 4000200, -1],  # exec buy +1
        [34203.0, 3, 4, 100, 4000000, 1],  # delete bid → -1
        [34204.0, 1, 5, 50, 4000050, -1],  # submission sell → -1
    ]
    rows = [
        [4000200, 50, 3999900, 400],  # imb = (400-50)/450 = 7/9
        [4000200, 40, 3999900, 400],  # imb = 360/440 = 9/11
        [4000200, 30, 3999900, 400],  # imb = 370/430 = 37/43
        [4000200, 30, 3999900, 300],  # imb = 270/330 = 9/11
        [4000200, 80, 3999900, 300],  # imb = 220/380 = 11/19
    ]
    msg, ob = _write_tape(tmp_path, events, rows)
    samples = _lobster_samples(msg, ob)
    assert [(round(i, 4), s, k) for i, s, k in samples] == [
        (round(7 / 9, 4), 1, "exec"),
        (round(9 / 11, 4), 1, "exec"),
        (round(37 / 43, 4), -1, "cancel"),
        (round(9 / 11, 4), -1, "submission"),
    ]


def test_measure_monotone_slope() -> None:
    # sign fully determined by imbalance sign → maximal positive gradient
    samples: list[tuple[float, int, str]] = [
        (i / 100.0, 1 if i >= 0 else -1, "exec") for i in range(-100, 101, 2)
    ]
    out = _measure(samples)
    assert out["ok"] is True
    qs = out["quintiles"]
    p_buys = [r["p_buy"] for r in qs]
    assert p_buys == sorted(p_buys)
    assert out["p_buy_slope_per_quintile"] > 0


def test_measure_iid_flat() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    imbs = rng.uniform(-1, 1, 5000)
    signs = rng.choice([-1, 1], 5000)
    samples = [(float(i), int(s), "exec") for i, s in zip(imbs, signs, strict=True)]
    out = _measure(samples)
    assert out["ok"] is True
    assert abs(out["p_buy_slope_per_quintile"]) < 0.05
    assert abs(out["overall_p_buy"] - 0.5) < 0.05
    # conditioned tables share pooled edges
    for kind in ("exec", "submission"):
        assert out["by_type"][kind]["n"] == out["n_by_type"][kind]


def test_measure_too_few() -> None:
    assert _measure([(0.5, 1, "exec")] * 10)["ok"] is False


def test_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        imbalance_predict_bench(tmp_path)


def _crafted_bench_tape(tmp_path: Path) -> None:
    # 61 events: alternating bid-heavy / ask-heavy books so both tails fill
    events: list[list[object]] = []
    rows: list[list[object]] = []
    for i in range(61):
        heavy_bid = i % 2 == 0
        b_sz, a_sz = (400, 100) if heavy_bid else (100, 400)
        # buy execs after bid-heavy rows, sell execs after ask-heavy rows
        events.append([34200.0 + i, 4, i + 1, 10, 4000000, -1 if i > 0 and heavy_bid else 1])
        rows.append([4000100, a_sz, 3999900, b_sz])
    _write_tape(tmp_path, events, rows)


def test_bench_determinism(tmp_path: Path) -> None:
    _crafted_bench_tape(tmp_path)
    first = imbalance_predict_bench(tmp_path)
    second = imbalance_predict_bench(tmp_path)
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert first == second


def test_receipt_seal_and_no_nan(tmp_path: Path) -> None:
    _crafted_bench_tape(tmp_path)
    payload = imbalance_predict_bench(tmp_path)
    seal = payload.pop("receipt_sha256")
    assert seal == hash_bytes(canonical_json_bytes(payload))
    # receipt must be strict-JSON clean (no NaN/Infinity survive)
    json.dumps(payload, allow_nan=False)


def test_sim_arm_measured() -> None:
    out = sim_imbalance_predict(horizon=3000, seed=3)
    assert out["ok"] is True
    assert out["mechanism_present"] is True
    assert out["n"] > 100
    assert out["n_by_type"]["exec"] > 0
