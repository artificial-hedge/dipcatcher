"""Unit tests for wave-307 regex-2 canon modules."""

from quant_fund.models.bitap_fuzzy import bitap_search
from quant_fund.models.glushkov_nfa import glushkov_match
from quant_fund.models.lazy_dfa import LazyDFA
from quant_fund.models.literal_prefilter import _literal_str
from quant_fund.models.pike_vm import _Parser
from quant_fund.models.pike_vm import compile as pike_compile
from quant_fund.models.pike_vm import run as pike_run
from quant_fund.models.regex_simplify import _serialize, simplify


def test_pike_captures():
    prog, ng = pike_compile(r"(a*)(b+)")
    got = pike_run(prog, ng, "aabbb")
    assert got is not None and got[1] == (0, 2) and got[2] == (2, 5)


def test_pike_no_match():
    prog, ng = pike_compile("abc")
    assert pike_run(prog, ng, "abd") is None


def test_lazy_dfa_memoizes():
    d = LazyDFA("(a|b)*abb")
    assert d.fullmatch("aabb")
    assert not d.fullmatch("aba")
    n0 = d.n_states()
    d.fullmatch("babb")
    assert d.n_states() <= n0 + 8


def test_bitap_exact():
    assert {e for e, _ in bitap_search("abc", "zabcz", 0)} == {4}


def test_literal_str():
    assert _literal_str(".*needle.*") == "needle"
    assert _literal_str("a|b") is None


def test_glushkov_match():
    assert glushkov_match("a?a?a?bbb", "bbb")
    assert not glushkov_match("(a|b)*abb", "ab")


def test_simplify_idempotent():
    ast = _Parser("ab|ac|a(d|d)").parse()[0][1]
    s1 = simplify(ast)
    assert _serialize(simplify(s1)) == _serialize(s1)
