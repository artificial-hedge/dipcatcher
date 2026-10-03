"""Tests for microstructure/sim_sensitivity.py — ZI-LOB response-surface map.

The lane maps the simulator's own flow parameters (lam, mu, theta_cxl) onto
emergent microstructure statistics under two flow arms (iid / two-state
Markov). Everything is a labeled SYNTHETIC correctness diagnostic of the
simulator — never market evidence.
"""

from __future__ import annotations

import json
import math

import pytest

from quant_fund.microstructure.sim_sensitivity import (
    ARMS,
    GRID_LAM,
    GRID_MU,
    GRID_THETA_CXL,
    STAT_NAMES,
    sensitivity_grid,
    sim_sensitivity_bench,
    sim_stats,
    write_sim_sensitivity_receipt,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, santa_fe_config
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

EXPECTED_STAT_KEYS = {*STAT_NAMES, "label", "n_trades"}


def _cfg(seed: int = 7) -> ZILobConfig:
    return santa_fe_config(seed=seed)


# ---------------------------------------------------------------------------
# sim_stats: schema + determinism
# ---------------------------------------------------------------------------


def test_sim_stats_schema_and_ranges() -> None:
    stats = sim_stats(_cfg(seed=7), None, 600)
    assert set(stats) == EXPECTED_STAT_KEYS
    assert stats["label"] == "SYNTHETIC"
    assert 0.0 <= stats["mo_fraction"] <= 1.0
    assert stats["n_trades"] > 0
    assert stats["n_trades"] == pytest.approx(stats["mo_fraction"] * 600)
    assert stats["sign_lag1"] is not None and -1.0 <= stats["sign_lag1"] <= 1.0
    assert stats["spread_ticks_median"] is not None and stats["spread_ticks_median"] >= 1.0
    assert stats["mid_move_std"] is not None and stats["mid_move_std"] >= 0.0
    assert stats["mean_top_depth"] is not None and stats["mean_top_depth"] > 0.0


def test_sim_stats_deterministic_same_seed() -> None:
    a = sim_stats(_cfg(seed=11), None, 500)
    b = sim_stats(_cfg(seed=11), None, 500)
    assert a == b
    c = sim_stats(_cfg(seed=12), None, 500)
    assert c != a  # a different seed traces a different path


def test_sim_stats_fail_closed() -> None:
    with pytest.raises(ValueError):
        sim_stats(_cfg(), None, 0)
    with pytest.raises(ValueError):
        sim_stats(_cfg(), None, -10)
    with pytest.raises(TypeError):
        sim_stats("not a config", None, 10)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Extreme / fail-closed parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"mu": 0.0}, {"lam": 0.0}, {"theta_cxl": 0.0}])
def test_extreme_zero_rates_fail_closed(kwargs: dict) -> None:
    # mu=0 would produce zero trades; the config refuses it fail-closed
    # rather than emitting a degenerate all-None stats row.
    with pytest.raises(ValueError):
        ZILobConfig(**kwargs)


def test_stats_degenerate_run_stays_honest() -> None:
    # A horizon too short for any trade leaves sign_lag1 honestly unmeasured.
    stats = sim_stats(_cfg(seed=3), None, 5)
    assert set(stats) == EXPECTED_STAT_KEYS
    for key in stats:
        value = stats[key]
        assert not (isinstance(value, float) and not math.isfinite(value))


# ---------------------------------------------------------------------------
# sensitivity_grid: tiny-horizon run
# ---------------------------------------------------------------------------


def test_sensitivity_grid_runs_and_maps_drivers() -> None:
    grid = sensitivity_grid(horizon=300, seed=5)
    assert grid["label"] == "SYNTHETIC"
    assert grid["horizon"] == 300
    assert grid["grid"]["lam"] == list(GRID_LAM)
    assert grid["grid"]["mu"] == list(GRID_MU)
    assert grid["grid"]["theta_cxl"] == list(GRID_THETA_CXL)
    cells = grid["cells"]
    assert len(cells) == len(GRID_LAM) * len(GRID_MU) * len(GRID_THETA_CXL)
    for cell in cells:
        assert set(cell) == {"lam", "mu", "theta_cxl", *ARMS}
        assert cell["lam"] in GRID_LAM
        assert cell["mu"] in GRID_MU
        assert cell["theta_cxl"] in GRID_THETA_CXL
        for arm in ARMS:
            assert set(cell[arm]) == EXPECTED_STAT_KEYS
    drivers = grid["drivers"]
    assert set(drivers) == set(ARMS)
    for arm in ARMS:
        for stat in STAT_NAMES:
            entry = drivers[arm][stat]
            assert set(entry) == {"param", "rho", "rhos", "n_points"}
            assert set(entry["rhos"]) == {"lam", "mu", "theta_cxl"}
            if entry["param"] is not None:
                assert entry["param"] in entry["rhos"]
                assert entry["rho"] == entry["rhos"][entry["param"]]
                rhos = [abs(r) for r in entry["rhos"].values() if r is not None]
                assert rhos and abs(entry["rho"]) == max(rhos)


def test_sensitivity_grid_deterministic() -> None:
    a = sensitivity_grid(horizon=200, seed=9)
    b = sensitivity_grid(horizon=200, seed=9)
    assert a == b


def test_mu_moves_mo_fraction_monotone() -> None:
    # The response surface's first-order check: mo_fraction must rise with mu
    # within each (lam, theta_cxl) slice of the iid arm.
    grid = sensitivity_grid(horizon=400, seed=4)
    cells = grid["cells"]
    for lam in GRID_LAM:
        for tc in GRID_THETA_CXL:
            fracs = [
                next(
                    c for c in cells if c["lam"] == lam and c["theta_cxl"] == tc and c["mu"] == mu
                )["iid"]["mo_fraction"]
                for mu in sorted(GRID_MU)
            ]
            assert fracs[0] < fracs[-1], f"mo_fraction not increasing in mu at lam={lam} tc={tc}"


# ---------------------------------------------------------------------------
# Sealed receipt
# ---------------------------------------------------------------------------


def test_bench_receipt_schema_and_seal() -> None:
    payload = sim_sensitivity_bench(horizon=200, seed=7)
    assert payload["kind"] == "sim_sensitivity"
    assert payload["schema"] == "sim_sensitivity.v1"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["research_only"] is True
    assert payload["git_revision"]
    seal = payload["receipt_sha256"]
    assert isinstance(seal, str) and len(seal) == 64
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    assert hash_bytes(canonical_json_bytes(body)) == seal
    # No NaN/Infinity anywhere — the strict verifier rejects non-finite leaves.
    assert json.dumps(payload, allow_nan=False)


def test_write_receipt_roundtrip(tmp_path) -> None:
    path = write_sim_sensitivity_receipt(tmp_path / "sim_sensitivity.json", horizon=200, seed=2)
    payload = json.loads(path.read_text())
    assert payload["receipt_sha256"]
    from quant_fund.research.receipt_v2 import verify_receipt_file

    result = verify_receipt_file(path)
    assert result["valid"], result["errors"]
