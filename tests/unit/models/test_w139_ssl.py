"""Wave-139 SSL + TTA canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._ssl_synth import linear_probe_acc, make_views, synth_ssl, synth_tta_split
from quant_fund.models.barlow_twins import bench_barlow_twins
from quant_fund.models.byol import bench_byol
from quant_fund.models.shot_tta import bench_shot_tta
from quant_fund.models.tent_tta import bench_tent_tta
from quant_fund.models.ttt_layer import bench_ttt_layer
from quant_fund.models.vicreg import bench_vicreg


class TestFixture:
    def test_ssl(self) -> None:
        x, y = synth_ssl(8, np.random.default_rng(0))
        assert x.shape == (8, 16) and y.shape == (8,)

    def test_views(self) -> None:
        x, _y = synth_ssl(8, np.random.default_rng(0))
        v1, v2 = make_views(x, np.random.default_rng(1))
        assert v1.shape == x.shape and v2.shape == x.shape

    def test_tta(self) -> None:
        xtr, ytr, xte, yte = synth_tta_split(8, 6, np.random.default_rng(0))
        assert xtr.shape == (8, 16) and xte.shape == (6, 16)
        assert linear_probe_acc(xtr, ytr, xte, yte) >= 0


class TestBYOL:
    def test_bench(self) -> None:
        out = bench_byol(seed=3, n_train=80, n_probe=40, iters=40)
        assert 0 <= out["synthetic_byol_probe_acc"] <= 1


class TestBarlow:
    def test_bench(self) -> None:
        out = bench_barlow_twins(seed=5, n_train=80, n_probe=40, iters=40)
        assert 0 <= out["synthetic_barlow_probe_acc"] <= 1


class TestVICReg:
    def test_bench(self) -> None:
        out = bench_vicreg(seed=7, n_train=80, n_probe=40, iters=40)
        assert 0 <= out["synthetic_vicreg_probe_acc"] <= 1


class TestTent:
    def test_bench(self) -> None:
        out = bench_tent_tta(seed=9, n_train=80, n_test=60, iters=40, tta_iters=20)
        assert 0 <= out["synthetic_tent_acc_after"] <= 1


class TestSHOT:
    def test_bench(self) -> None:
        out = bench_shot_tta(seed=11, n_train=80, n_test=60, iters=40, tta_iters=20)
        assert 0 <= out["synthetic_shot_acc_after"] <= 1


class TestTTT:
    def test_bench(self) -> None:
        out = bench_ttt_layer(seed=13, n_train=80, n_test=60, iters=40, ttt_iters=20)
        assert 0 <= out["synthetic_ttt_acc_after"] <= 1
