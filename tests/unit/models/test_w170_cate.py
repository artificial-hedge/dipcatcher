"""Wave-170 causal-DL-2 canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cate_distill import bench_cate_distill
from quant_fund.models.causal_rep_bal import bench_causal_rep_bal
from quant_fund.models.net_drlearner import bench_net_drlearner
from quant_fund.models.rlearner import bench_rlearner
from quant_fund.models.slearner_tlearner import bench_slearner_tlearner
from quant_fund.models.xlearner import bench_xlearner


class TestXLearner:
    def test_bench(self) -> None:
        out = bench_xlearner(seed=3, iters=120)
        assert np.isfinite(out["synthetic_xlearner_pehe"])


class TestRLearner:
    def test_bench(self) -> None:
        out = bench_rlearner(seed=5, iters=150)
        assert np.isfinite(out["synthetic_rlearner_pehe"])


class TestSTLearner:
    def test_bench(self) -> None:
        out = bench_slearner_tlearner(seed=7, iters=120)
        assert np.isfinite(out["synthetic_st_s_pehe"])


class TestCRB:
    def test_bench(self) -> None:
        out = bench_causal_rep_bal(seed=9, iters=120)
        assert np.isfinite(out["synthetic_crb_pehe"])


class TestCateDistill:
    def test_bench(self) -> None:
        out = bench_cate_distill(seed=11)
        assert np.isfinite(out["synthetic_cd_pehe"])


class TestNetDR:
    def test_bench(self) -> None:
        out = bench_net_drlearner(seed=13, iters=120)
        assert np.isfinite(out["synthetic_ndr_pehe"])
