"""SYNTHETIC inverted index + boolean retrieval.

term → sorted doc-id postings; AND/OR/NOT evaluation via merge walks,
verified against brute-force document scan.
"""

from __future__ import annotations

import random


def build_index(docs: list[list[str]]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for d, doc in enumerate(docs):
        for t in set(doc):
            idx.setdefault(t, []).append(d)
    return idx


def _merge_and(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def _merge_or(a: list[int], b: list[int]) -> list[int]:
    out = sorted(set(a) | set(b))
    return out


def _diff(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) - set(b))


def query(idx: dict[str, list[int]], q: tuple, ndocs: int) -> list[int]:
    op, *args = q
    if op == "term":
        return idx.get(args[0], [])
    if op == "and":
        return _merge_and(query(idx, args[0], ndocs), query(idx, args[1], ndocs))
    if op == "or":
        return _merge_or(query(idx, args[0], ndocs), query(idx, args[1], ndocs))
    if op == "not":
        return _diff(list(range(ndocs)), query(idx, args[0], ndocs))
    raise ValueError(op)


def _brute(docs: list[list[str]], q: tuple) -> list[int]:
    op, *args = q
    if op == "term":
        return [d for d, doc in enumerate(docs) if args[0] in doc]
    if op == "and":
        return sorted(set(_brute(docs, args[0])) & set(_brute(docs, args[1])))
    if op == "or":
        return sorted(set(_brute(docs, args[0])) | set(_brute(docs, args[1])))
    if op == "not":
        return sorted(set(range(len(docs))) - set(_brute(docs, args[0])))
    raise ValueError(op)


def bench_inverted_index(seed: int = 20261231 + 460) -> dict[str, float]:
    rng = random.Random(seed)
    vocab = [f"t{i}" for i in range(30)]
    match = cover = 0
    trials = 40
    for _ in range(trials):
        docs = [rng.sample(vocab, rng.randrange(1, 10)) for _ in range(rng.randrange(5, 20))]
        idx = build_index(docs)
        t1, t2, t3 = rng.sample(vocab, 3)
        q = (
            "and",
            ("term", t1),
            (
                "or",
                ("term", t2),
                (
                    "not",
                    ("term", t3),
                ),
            ),
        )
        match += int(query(idx, q, len(docs)) == _brute(docs, q))
        cover += int(len(set().union(*(set(idx.get(t, [])) for t in vocab))) == len(docs))
    return {
        "synthetic_query_matches_brute": float(match / trials),
        "synthetic_postings_cover_corpus": float(cover / trials),
    }
