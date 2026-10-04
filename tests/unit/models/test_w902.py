"""Wave-902 trie/string-index canon tests."""

from __future__ import annotations

from quant_fund.models.crit_bit_tree import bench_crit_bit_tree
from quant_fund.models.patricia_trie import bench_patricia_trie
from quant_fund.models.radix_trie import bench_radix_trie
from quant_fund.models.suffix_trie import bench_suffix_trie
from quant_fund.models.ternary_trie import bench_ternary_trie
from quant_fund.models.trie import bench_trie


def test_trie():
    assert bench_trie()["synthetic_trie"] == 1.0


def test_patricia_trie():
    assert bench_patricia_trie()["synthetic_patricia_trie"] == 1.0


def test_suffix_trie():
    assert bench_suffix_trie()["synthetic_suffix_trie"] == 1.0


def test_ternary_trie():
    assert bench_ternary_trie()["synthetic_ternary_trie"] == 1.0


def test_radix_trie():
    assert bench_radix_trie()["synthetic_radix_trie"] == 1.0


def test_crit_bit_tree():
    assert bench_crit_bit_tree()["synthetic_crit_bit_tree"] == 1.0
