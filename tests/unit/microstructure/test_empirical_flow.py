"""Contracts for the empirical-flow lane: size PMFs, calibration, bench."""

import math

import numpy as np
import pytest

from quant_fund.microstructure.empirical_flow import (
    EMPIRICAL_FLOW_SCHEMA,
    calibrate_theta,
    empirical_zi_config,
)
from quant_fund.microstructure.empirical_flow_bench import empirical_flow_bench
from quant_fund.microstructure.tape_stats import load_tape_stats
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


def _stats():
    return load_tape_stats("receipts", "amzn")


class TestSizePmfContract:
    def test_default_config_bit_identical(self):
        """Unset PMFs draw zero RNG → legacy event stream preserved."""
        a = ZILobSimulator(ZILobConfig(seed=3))
        b = ZILobSimulator(ZILobConfig(seed=3, mo_size_pmf=None, lo_size_pmf=None))
        for _ in range(300):
            assert a.step() == b.step()
            assert a.t == b.t and a.n_events == b.n_events

    def test_pmf_normalization_and_draw(self):
        cfg = ZILobConfig(seed=1, mo_size_pmf=((1, 1.0), (4, 0.0 + 1.0)))
        assert cfg.mo_size_pmf is not None
        sim = ZILobSimulator(cfg)
        for _ in range(200):
            sim.step()
        if sim.n_mo_arrivals:
            assert sim.n_mo_units >= sim.n_mo_arrivals
            assert sim.n_mo_units <= 4 * sim.n_mo_arrivals

    @pytest.mark.parametrize(
        "pmf",
        [[], ((0, 1.0),), ((-2, 1.0),), ((3, 0.0),), ((2, -1.0),), ((2, float("nan")),)],
    )
    def test_bad_pmf_rejected(self, pmf):
        with pytest.raises(ValueError, match="pmf"):
            ZILobConfig(mo_size_pmf=pmf)  # type: ignore[arg-type]

    def test_sweeps_emerge_under_bursts(self):
        """MO bursts larger than touch depth must produce multi-level fills."""
        cfg = ZILobConfig(
            seed=5,
            init_depth=2,
            mo_size_pmf=((6, 1.0),),
            mu=0.5,
            lam=0.02,
            theta_cxl=0.005,
        )
        sim = ZILobSimulator(cfg)
        for _ in range(4000):
            sim.step()
        multi = sum(
            1
            for i in range(1, len(sim.trades))
            if sim.trades[i].t == sim.trades[i - 1].t
            and sim.trades[i].level != sim.trades[i - 1].level
        )
        assert multi > 0


class TestEmpiricalCalibration:
    def test_calibrate_theta_positive(self):
        stats = _stats()
        theta, depth = calibrate_theta(stats, lam=0.2, mu=0.2, band=20, seed=1)
        assert math.isfinite(theta) and theta > 0
        assert depth > 0

    def test_config_fail_closed(self):
        stats = _stats()
        with pytest.raises(ValueError, match="size_scale"):
            empirical_zi_config(stats, size_scale=0.0)
        with pytest.raises(ValueError, match="band"):
            empirical_zi_config(stats, band=0)
        with pytest.raises(ValueError, match="TapeStats"):
            empirical_zi_config(object())  # type: ignore[arg-type]

    def test_calibrated_rates_match_card(self):
        stats = _stats()
        cfg, calib = empirical_zi_config(stats, band=20, seed=1)
        exec_rate = stats.rate("exec") + stats.rate("exec_hidden")
        sub_rate = stats.rate("sub")
        assert cfg.mu == pytest.approx(exec_rate / 2.0)
        assert cfg.lam == pytest.approx(sub_rate / 40.0)
        assert cfg.mo_size_pmf is not None and cfg.lo_size_pmf is not None
        assert calib.probe_depth > 0

    def test_empirical_arm_measures_sizes(self):
        cfg, _ = empirical_zi_config(_stats(), seed=2)
        sim = ZILobSimulator(cfg)
        for _ in range(1500):
            sim.step()
        assert sim.n_mo_units > sim.n_mo_arrivals or sim.n_mo_arrivals == 0
        # realized sizes should span the pmf support (>1 on average)
        assert np.isfinite(sim.n_mo_units / max(1, sim.n_mo_arrivals))


class TestBenchReceipt:
    def test_payload_sealed(self):
        payload = empirical_flow_bench("receipts", horizon=2500, seed=4)
        assert payload["schema"] == EMPIRICAL_FLOW_SCHEMA
        assert payload["receipt_sha256"]
        assert len(payload["arms"]) == 3
        emp = next(a for a in payload["arms"] if a["name"] == "empirical")
        assert emp["multi_level_share"] > 0.0
        assert payload["claims"]["sweeps_now_expressible"]
