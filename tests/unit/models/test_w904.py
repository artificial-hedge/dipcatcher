"""Wave-904 balanced-tree canon tests."""

from __future__ import annotations

from quant_fund.models.aa_tree import bench_aa_tree
from quant_fund.models.avl_tree import bench_avl_tree
from quant_fund.models.red_black_tree import bench_red_black_tree
from quant_fund.models.scapegoat_tree import bench_scapegoat_tree
from quant_fund.models.splay_tree import bench_splay_tree
from quant_fund.models.treap import bench_treap


def test_avl_tree():
    assert bench_avl_tree()["synthetic_avl_tree"] == 1.0


def test_red_black_tree():
    assert bench_red_black_tree()["synthetic_red_black_tree"] == 1.0


def test_splay_tree():
    assert bench_splay_tree()["synthetic_splay_tree"] == 1.0


def test_treap():
    assert bench_treap()["synthetic_treap"] == 1.0


def test_scapegoat_tree():
    assert bench_scapegoat_tree()["synthetic_scapegoat_tree"] == 1.0


def test_aa_tree():
    assert bench_aa_tree()["synthetic_aa_tree"] == 1.0
