"""Wave-189 self-play/game-AI canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.alphazero_lite import bench_alphazero_lite
from quant_fund.models.deep_cfr import bench_deep_cfr
from quant_fund.models.expert_iteration import bench_expert_iteration
from quant_fund.models.mccfr_outcome import bench_mccfr_outcome
from quant_fund.models.nfsp import bench_nfsp
from quant_fund.models.psro import bench_psro


class TestAZ:
    def test_bench(self) -> None:
        out = bench_alphazero_lite(seed=3, episodes=8, sims=8)
        assert 0.0 <= out["synthetic_az_nonloss_random"] <= 1.0


class TestExIt:
    def test_bench(self) -> None:
        out = bench_expert_iteration(seed=5, rounds=5)
        assert 0.0 <= out["synthetic_exit_nonloss_oracle"] <= 1.0


class TestNFSP:
    def test_bench(self) -> None:
        out = bench_nfsp(seed=7, iters=10)
        assert np.isfinite(out["synthetic_nfsp_expl"])


class TestPSRO:
    def test_bench(self) -> None:
        out = bench_psro(seed=9, epochs=2, games=4)
        assert np.isfinite(out["synthetic_psro_nonloss_oracle"])


class TestDCFR:
    def test_bench(self) -> None:
        out = bench_deep_cfr(seed=11, iters=40)
        assert np.isfinite(out["synthetic_dcfr_expl"])
        assert np.isfinite(out["synthetic_amortize_gap"])


class TestMCCFR:
    def test_bench(self) -> None:
        out = bench_mccfr_outcome(seed=13, iters=500)
        assert np.isfinite(out["synthetic_mccfr_expl"])
