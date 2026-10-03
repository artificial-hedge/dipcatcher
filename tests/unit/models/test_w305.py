"""Unit tests for wave-305 text-index-2/stringology canon modules."""

from quant_fund.models.booth_rotation import booth
from quant_fund.models.lyndon_factor import duval
from quant_fund.models.palindromic_tree import build_eertree
from quant_fund.models.suffix_array_lcp import kasai_lcp, pattern_bounds, suffix_array
from quant_fund.models.suffix_tree_lex import build_suffix_tree, contains, occurrences
from quant_fund.models.z_function import z_function, z_search


def test_sa_sorted():
    s = "banana"
    sa = suffix_array(s)
    assert [s[i:] for i in sa] == sorted(s[i:] for i in range(6))
    lcp = kasai_lcp(s, sa)
    assert lcp[sa.index(1)] == 3  # 'ana' vs 'anana'


def test_pattern_bounds():
    s = "ababab"
    sa = suffix_array(s)
    lo, hi = pattern_bounds(s, sa, "ab")
    assert sorted(sa[i] for i in range(lo, hi)) == [0, 2, 4]


def test_z_function():
    assert z_function("aabcaab") == [7, 1, 0, 0, 3, 1, 0]
    assert z_search("abcabc", "bc") == [1, 4]


def test_suffix_tree():
    s = "banana"
    root = build_suffix_tree(s)
    assert contains(root, s, "nan")
    assert not contains(root, s, "xyz")
    assert occurrences(root, s, "an") == [1, 3]


def test_booth():
    assert booth("bbaa") == 2
    assert booth("aaaa") == 0


def test_duval_lyndon():
    parts = duval("abcabc")
    assert "".join(parts) == "abcabc"


def test_eertree_counts():
    nodes, _ = build_eertree("aaa")
    assert len(nodes) - 2 == 3
