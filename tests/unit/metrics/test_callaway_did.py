"""Tests for metrics/callaway_did.py — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.callaway_did import (
    agg_att,
    att_gt,
    bench_callaway_did,
    boot_se,
    pretrend_test,
    synth_panel,
)


class TestAttGt:
    def test_structure(self) -> None:
        data, _ = synth_panel(n_units=200, periods=8, seed=0)
        res = att_gt(data)
        assert res.att.size > 0
        assert res.att.size == res.se.size == res.g.size == res.t.size
        assert res.pretrend.any()  # pre-period cells exist
        assert res.att.size == res.pretrend.size

    def test_recovers_effect(self) -> None:
        data, true_att = synth_panel(n_units=400, periods=8, seed=1)
        res = att_gt(data)
        agg = agg_att(res, "simple")
        true_simple = float(np.mean(list(true_att.values())))
        assert abs(agg.value[0] - true_simple) < 0.3

    def test_pretrend_cells_small_on_parallel_trends(self) -> None:
        data, _ = synth_panel(n_units=400, periods=8, seed=2)
        res = att_gt(data)
        # pseudo-ATTs pre-treatment should be ~0 when parallel trends hold
        assert np.abs(res.att[res.pretrend]).mean() < 0.4

    def test_no_never_treated_raises(self) -> None:
        data, _ = synth_panel(n_units=200, periods=8, seed=3)
        data["g"] = np.where(data["g"] == 0, 4, data["g"])  # all treated
        with pytest.raises(ValueError):
            att_gt(data, control="never")

    def test_no_treated_raises(self) -> None:
        data, _ = synth_panel(n_units=200, periods=8, seed=4)
        data["g"] = np.zeros_like(data["g"])
        with pytest.raises(ValueError):
            att_gt(data)

    def test_invalid(self) -> None:
        data, _ = synth_panel(n_units=100, periods=6, seed=5)
        with pytest.raises(ValueError):
            att_gt(data, control="bogus")
        with pytest.raises(ValueError):
            att_gt({"unit": np.array([1.0])})
        with pytest.raises(ValueError):
            att_gt(
                {
                    "unit": np.array([1.0, 2.0]),
                    "g": np.array([0, 1]),
                    "t": np.array([0, 0]),
                    "y": np.array([np.nan, 1.0]),
                }
            )


class TestAgg:
    def test_kinds(self) -> None:
        data, _ = synth_panel(n_units=200, periods=8, seed=6)
        res = att_gt(data)
        for kind in ("simple", "calendar", "event", "group"):
            agg = agg_att(res, kind)
            assert np.all(np.isfinite(agg.value))
        e = agg_att(res, "event")
        assert np.all(e.keys >= 0)  # only post-treatment event times

    def test_invalid(self) -> None:
        data, _ = synth_panel(n_units=100, periods=6, seed=7)
        res = att_gt(data)
        with pytest.raises(ValueError):
            agg_att(res, "bogus")


class TestInference:
    def test_boot_se_positive(self) -> None:
        data, _ = synth_panel(n_units=200, periods=8, seed=8)
        se = boot_se(data, n_boot=25, seed=0)
        assert np.all(se > 0)

    def test_pretrend(self) -> None:
        data, _ = synth_panel(n_units=400, periods=8, seed=9)
        _c, p = pretrend_test(att_gt(data))
        assert p > 0.05  # parallel trends hold → no rejection

    def test_pretrend_rejects_when_planted(self) -> None:
        data, _ = synth_panel(n_units=400, periods=8, pretrend_slope=1.5, seed=10)
        _c, p = pretrend_test(att_gt(data))
        assert p < 0.05

    def test_pretrend_invalid(self) -> None:
        data, _ = synth_panel(
            n_units=100, periods=6, treat_periods=(3,), treat_frac=(0.3,), seed=11
        )
        res = att_gt(data)
        # strip pre-period cells to force the error path
        keep = ~res.pretrend
        res_post = type(res)(
            g=res.g[keep],
            t=res.t[keep],
            att=res.att[keep],
            se=res.se[keep],
            n=res.n[keep],
            pretrend=res.pretrend[keep],
        )
        with pytest.raises(ValueError):
            pretrend_test(res_post)


class TestBench:
    def test_keys_finite(self) -> None:
        blob = bench_callaway_did()
        assert len(blob) >= 8
        assert all(np.isfinite(v) for v in blob.values())
        for key in blob:
            assert key.startswith("synthetic_")

    def test_bench_quality(self) -> None:
        blob = bench_callaway_did()
        assert blob["synthetic_att_bias"] < 0.4
        assert blob["synthetic_pretrend_p"] > 0.05
        assert blob["synthetic_pretrend_reject_p"] < 0.05
        assert blob["synthetic_determinism"] == 1.0
