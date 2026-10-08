"""Synthetic corpus for the w145 retrieval canon (SYNTHETIC).

Documents = bags of tokens drawn from hidden topic mixtures; queries
sample a topic. Semantic relevance = shared dominant topic (lexical
overlap is partial — synonyms alias the same topic).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_VOCAB = 500
_NTOPIC = 8
_SYN_GROUPS = 40  # groups of synonym tokens sharing meaning


def _topic_tokens(rng: np.random.Generator) -> NDArray[np.int64]:
    return np.arange(_NTOPIC)[:, None] * (_VOCAB // _NTOPIC) + np.arange(_VOCAB // _NTOPIC)[None, :]


def synth_corpus(
    n_docs: int,
    doc_len: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """docs (n,doc_len) token ids; topics (n,) dominant topic id."""
    tt = _topic_tokens(rng)
    topics = rng.integers(0, _NTOPIC, n_docs)
    docs = np.zeros((n_docs, doc_len), dtype=np.int64)
    for i in range(n_docs):
        tp = tt[topics[i]]
        docs[i] = rng.choice(tp, doc_len)
        # 20% background noise tokens
        noise = rng.random(doc_len) < 0.2
        docs[i][noise] = rng.integers(0, _VOCAB, noise.sum())
    return docs, topics


def synth_queries(
    n_q: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """queries are short strings of the topic's tokens (lexically weak)."""
    tt = _topic_tokens(rng)
    topics = rng.integers(0, _NTOPIC, n_q)
    qs = np.zeros((n_q, 4), dtype=np.int64)
    for i in range(n_q):
        qs[i] = rng.choice(tt[topics[i]], 4)
        # 70% of query tokens are synonyms: shifted aliases (no lexical match)
        alias = rng.random(4) < 0.7
        qs[i][alias] = (qs[i][alias] + _VOCAB // 2) % _VOCAB
    return qs, topics


def recall_at_k(scores: FloatArray, topics_d, topics_q, k: int) -> float:
    top = scores.argsort(-1)[:, -k:]
    hit = (topics_d[top] == topics_q[:, None]).any(-1)
    return float(hit.mean())
