"""Wave-160 federated-optimization canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.ditto_fl import bench_ditto_fl
from quant_fund.models.fednova_fl import bench_fednova_fl
from quant_fund.models.fedopt_adam import bench_fedopt_adam
from quant_fund.models.mime_lite import bench_mime_lite
from quant_fund.models.moon_fl import bench_moon_fl
from quant_fund.models.scaffold_fl import bench_scaffold_fl


class TestScaffold:
    def test_bench(self) -> None:
        out = bench_scaffold_fl(seed=3, T=60, rounds=4, epochs=2)
        assert np.isfinite(out["synthetic_scaffold_err"])


class TestFedNova:
    def test_bench(self) -> None:
        out = bench_fednova_fl(seed=5, T=60, rounds=4)
        assert np.isfinite(out["synthetic_fednova_err"])


class TestDitto:
    def test_bench(self) -> None:
        out = bench_ditto_fl(seed=7, T=60, rounds=4, epochs=2)
        assert np.isfinite(out["synthetic_ditto_mse"])


class TestMoon:
    def test_bench(self) -> None:
        out = bench_moon_fl(seed=9, n_clients=2, rounds=2, local_ep=3, n=40)
        assert 0 <= out["synthetic_moon_acc"] <= 1


class TestFedOpt:
    def test_bench(self) -> None:
        out = bench_fedopt_adam(seed=11, T=60, rounds=4, epochs=2)
        assert np.isfinite(out["synthetic_fedopt_err"])


class TestMime:
    def test_bench(self) -> None:
        out = bench_mime_lite(seed=13, T=60, rounds=4, epochs=2)
        assert np.isfinite(out["synthetic_mime_err"])
