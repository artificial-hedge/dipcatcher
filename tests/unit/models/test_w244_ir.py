"""Wave-244 IR canon tests."""

from quant_fund.models.inverted_index import bench_inverted_index, build_index, query
from quant_fund.models.lsh_dedup import bench_lsh_dedup
from quant_fund.models.ngram_spell import _bigrams, bench_ngram_spell, suggest
from quant_fund.models.positional_index import bench_positional_index, build_positional, phrase_docs
from quant_fund.models.posting_merge import bench_posting_merge, galloping_intersect, skip_merge
from quant_fund.models.wand_bmw import bench_wand_bmw, wand_topk


def test_inverted_basic():
    idx = build_index([["a", "b"], ["b", "c"], ["a"]])
    assert query(idx, ("and", ("term", "a"), ("term", "b")), 3) == [0]
    assert query(idx, ("or", ("term", "a"), ("term", "c")), 3) == [0, 1, 2]


def test_inverted_bench():
    assert bench_inverted_index()["synthetic_query_matches_brute"] == 1.0


def test_posting_basic():
    a, b = [1, 3, 5, 9], [1, 2, 3, 4, 5]
    assert skip_merge(a, b) == [1, 3, 5]
    assert galloping_intersect(a, b) == [1, 3, 5]


def test_posting_bench():
    assert bench_posting_merge()["synthetic_skip_merge_exact"] == 1.0


def test_wand_basic():
    postings = {"x": [0, 1, 2], "y": [1, 3]}
    tfs = {"x": {0: 1, 1: 5, 2: 2}, "y": {1: 5, 3: 1}}
    assert wand_topk(postings, tfs, ["x", "y"], 1) == [1]


def test_wand_bench():
    assert bench_wand_bmw()["synthetic_wand_topk_exact"] == 1.0


def test_lsh_bench():
    out = bench_lsh_dedup()
    assert out["synthetic_duplicate_recall"] > 0.5


def test_spell_basic():
    vocab = ["portfolio", "volatility"]
    grams = {}
    for i, w in enumerate(vocab):
        for g in _bigrams(w):
            grams.setdefault(g, set()).add(i)
    assert suggest("portfoloi", vocab, grams) == "portfolio"


def test_spell_bench():
    assert bench_ngram_spell()["synthetic_top1_accuracy"] == 1.0


def test_positional_basic():
    docs = [["buy", "low", "sell", "high"], ["sell", "low"]]
    idx = build_positional(docs)
    assert phrase_docs(idx, ["buy", "low"]) == [0]
    assert phrase_docs(idx, ["low", "sell"]) == [0]


def test_positional_bench():
    assert bench_positional_index()["synthetic_phrase_matches_brute"] == 1.0
