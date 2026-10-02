"""Contract tests for place_law_bench — placement-law family coverage."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.place_law_bench import (
    PLACE_LAW_SCHEMA,
    lobster_place_law,
    place_law_bench,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.research.receipt_v2 import verify_receipt_file


def _write_tape(tmp_path: Path, n: int = 400) -> tuple[Path, Path]:
    """Synthetic LOBSTER pair: execs balanced buy/sell, subs at mid gap."""
    msg = tmp_path / "X_2020-01-01_0_1_message_10.csv"
    ob = tmp_path / "X_2020-01-01_0_1_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        mw, ow = csv.writer(fm), csv.writer(fo)
        for i in range(n):
            if i % 10 == 4:
                mw.writerow([i + 1.0, 4, i, 25, 1050000, -1])
                bid, ask = 1040000, 1060000
            elif i % 10 == 9:
                mw.writerow([i + 1.0, 4, i, 25, 1040000, 1])
                bid, ask = 1040000, 1060000
            else:
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


def test_lobster_place_law_shape(tmp_path: Path) -> None:
    msg, ob = _write_tape(tmp_path, n=200)
    r = lobster_place_law(msg, ob)
    assert r["ok"] is True
    assert r["n_submissions"] > 0
    assert r["all"]["ok"] is True
    assert len(r["all"]["hist"]) == 10
    assert r["all"]["mean"] is not None
    assert r["baseline"]["n"] >= 0


def test_place_mode_frac_cdf_hump() -> None:
    """The Binomial law peaks mid-band; frac=0 keeps the legacy CDF."""
    cfg = replace(santa_fe_config(seed=1), band=21, place_mode_frac=0.4)
    sim = ZILobSimulator(cfg)
    pmf = np.diff(np.concatenate([[0.0], sim._dist_cdf]))  # noqa: SLF001
    assert int(np.argmax(pmf)) + 1 == 9
    a = ZILobSimulator(santa_fe_config(seed=5))
    b = ZILobSimulator(replace(santa_fe_config(seed=5), place_mode_frac=0.0))
    np.testing.assert_array_equal(a._dist_cdf, b._dist_cdf)  # noqa: SLF001


def test_place_law_bench_seals(tmp_path: Path) -> None:
    _write_tape(tmp_path)
    payload = place_law_bench(tmp_path, "X", horizon=300, scan_horizon=200, seed=3)
    assert payload["schema"] == PLACE_LAW_SCHEMA
    assert set(payload["sim_arms"]) == {"iid", "lv_cd300_tilt", "lv_cd300_tilt_hump"}
    assert len(payload["power_scan"]) == 20
    assert len(payload["binom_scan"]) == 12
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload))
    result = verify_receipt_file(receipt)
    assert result.get("valid") is True


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        place_law_bench(tmp_path, "AMZN")


def test_dist_sim_ignores_seed_book() -> None:
    from quant_fund.microstructure.place_law_bench import _DistSim

    sim = _DistSim(santa_fe_config(seed=5), None)
    assert sim.dist_log == []
    sim.step()
