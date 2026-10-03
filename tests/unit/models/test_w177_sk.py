"""Wave-177 neuromorphic canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.izhikevich import bench_izhikevich
from quant_fund.models.lif_neuron import bench_lif_neuron
from quant_fund.models.lsm_reservoir import bench_lsm_reservoir
from quant_fund.models.stdp_learn import bench_stdp_learn
from quant_fund.models.surrogate_snn import bench_surrogate_snn
from quant_fund.models.temporal_code import bench_temporal_code


class TestLIF:
    def test_bench(self) -> None:
        out = bench_lif_neuron(seed=3)
        assert 0.0 <= out["synthetic_lif_acc"] <= 1.0


class TestSTDP:
    def test_bench(self) -> None:
        out = bench_stdp_learn(seed=5, T=200)
        assert out["synthetic_stdp_selectivity"] > 0.0


class TestSNN:
    def test_bench(self) -> None:
        out = bench_surrogate_snn(seed=7, iters=80, T=10)
        assert 0.0 <= out["synthetic_snn_acc"] <= 1.0


class TestIzh:
    def test_bench(self) -> None:
        out = bench_izhikevich(seed=9)
        assert out["synthetic_izh_rate_high"] >= out["synthetic_izh_rate_low"]


class TestLSM:
    def test_bench(self) -> None:
        out = bench_lsm_reservoir(seed=11, R=32, T=10)
        assert np.isfinite(out["synthetic_lsm_gap"])


class TestTemporal:
    def test_bench(self) -> None:
        out = bench_temporal_code(seed=13, T=15)
        assert out["synthetic_latency_spikes_per_input"] == 1.0
