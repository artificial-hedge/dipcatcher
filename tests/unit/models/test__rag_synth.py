"""Probes for _rag_synth."""

import numpy as np
import pytest

from quant_fund.models._rag_synth import recall_at_k, synth_corpus, synth_queries


def test_corpus_shapes_and_topic_alignment():
    rng = np.random.default_rng(0)
    docs, topics = synth_corpus(30, 20, rng)
    assert docs.shape == (30, 20) and topics.shape == (30,)
    assert topics.min() >= 0 and topics.max() < 8
    # docs of topic t should draw mostly from topic-t token band
    band = 500 // 8
    t0_docs = docs[topics == 0]
    frac_in_band = ((t0_docs >= 0) & (t0_docs < band)).mean()
    assert frac_in_band > 0.6  # 80% topical minus noise


def test_corpus_deterministic():
    a = synth_corpus(10, 8, np.random.default_rng(1))
    b = synth_corpus(10, 8, np.random.default_rng(1))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("kw", [{"n_docs": 0}, {"doc_len": 0}, {"n_docs": -2}])
def test_corpus_hostile(kw):
    with pytest.raises(ValueError):
        synth_corpus(kw.get("n_docs", 5), kw.get("doc_len", 5), np.random.default_rng(0))


def test_queries_shape_and_determinism():
    q1, t1 = synth_queries(12, np.random.default_rng(2))
    q2, t2 = synth_queries(12, np.random.default_rng(2))
    assert q1.shape == (12, 4)
    np.testing.assert_array_equal(q1, q2)


def test_queries_rejects_empty():
    with pytest.raises(ValueError):
        synth_queries(0, np.random.default_rng(0))


def test_recall_at_k_perfect_and_floor():
    rng = np.random.default_rng(0)
    topics_d = rng.integers(0, 8, 100)
    topics_q = np.array([0, 1])
    scores = np.tile(topics_d == 0, (2, 1)).astype(float)
    scores[1] = topics_d == 1
    # score ranks same-topic docs highest → recall@5 = 1
    assert recall_at_k(scores, topics_d, topics_q, 5) == 1.0
    # adversarial: query topic absent → recall 0
    tq2 = np.array([7])
    td2 = np.zeros(10, dtype=np.int64)
    s2 = np.random.default_rng(1).standard_normal((1, 10))
    assert recall_at_k(s2, td2, tq2, 3) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_recall_k_vacuous(k):
    # k<=0 previously returned 1.0 by selecting ALL docs — dishonest perfect recall
    s = np.zeros((2, 10))
    td = np.zeros(10, dtype=np.int64)
    tq = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError):
        recall_at_k(s, td, tq, k)


def test_recall_k_exceeds_docs():
    s = np.zeros((2, 10))
    td = np.zeros(10, dtype=np.int64)
    tq = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError):
        recall_at_k(s, td, tq, 11)


def test_recall_shape_mismatch():
    s = np.zeros((3, 10))
    td = np.zeros(9, dtype=np.int64)  # 9 != 10
    tq = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError):
        recall_at_k(s, td, tq, 5)
