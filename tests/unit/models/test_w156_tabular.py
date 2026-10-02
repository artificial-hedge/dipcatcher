"""Wave-156 tabular-DL + tokenizer canon tests."""

from __future__ import annotations

from quant_fund.models.grownet_boost import bench_grownet_boost
from quant_fund.models.node_net import bench_node_net
from quant_fund.models.soft_tree import bench_soft_tree
from quant_fund.models.tabm_mini import bench_tabm_mini
from quant_fund.models.tabular_resnet import bench_tabular_resnet
from quant_fund.models.tokenizer_bpe import _apply, _bpe_train, bench_tokenizer_bpe


class TestBPE:
    def test_merge(self) -> None:
        merges = _bpe_train([["ab", "ab", "abc"]], 3)
        assert merges

    def test_apply(self) -> None:
        assert "".join(_apply("ab", [("a", "b")])) == "ab"

    def test_bench(self) -> None:
        out = bench_tokenizer_bpe(seed=3, n_docs=30, n_merges=10)
        assert out["synthetic_bpe_compression"] >= 1


class TestTRes:
    def test_bench(self) -> None:
        out = bench_tabular_resnet(seed=5, n=120, iters=15)
        assert 0 <= out["synthetic_tres_acc"] <= 1


class TestNode:
    def test_bench(self) -> None:
        out = bench_node_net(seed=7, n=120, iters=15, depth=2)
        assert 0 <= out["synthetic_node_acc"] <= 1


class TestGN:
    def test_bench(self) -> None:
        out = bench_grownet_boost(seed=9, n=120, n_stages=2, iters=15)
        assert 0 <= out["synthetic_gn_acc"] <= 1


class TestST:
    def test_bench(self) -> None:
        out = bench_soft_tree(seed=11, n=120, iters=15, depth=2)
        assert 0 <= out["synthetic_st_acc"] <= 1


class TestTabM:
    def test_bench(self) -> None:
        out = bench_tabm_mini(seed=13, n=120, iters=15, k=4)
        assert 0 <= out["synthetic_tabm_acc"] <= 1
