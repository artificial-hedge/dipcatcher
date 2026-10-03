"""SYNTHETIC positional inverted index for phrase queries.

term → {doc: [positions]}; phrase match via positional intersection,
verified against brute-force token-sequence scan.
"""

from __future__ import annotations

import random


def build_positional(docs: list[list[str]]) -> dict[str, dict[int, list[int]]]:
    idx: dict[str, dict[int, list[int]]] = {}
    for d, doc in enumerate(docs):
        for p, t in enumerate(doc):
            idx.setdefault(t, {}).setdefault(d, []).append(p)
    return idx


def phrase_docs(idx: dict[str, dict[int, list[int]]], phrase: list[str]) -> list[int]:
    if not phrase:
        return []
    first = idx.get(phrase[0], {})
    out = []
    for d, poses in first.items():
        for p0 in poses:
            ok = True
            for off, t in enumerate(phrase[1:], 1):
                if p0 + off not in idx.get(t, {}).get(d, []):
                    ok = False
                    break
            if ok:
                out.append(d)
                break
    return sorted(set(out))


def _brute(docs: list[list[str]], phrase: list[str]) -> list[int]:
    n = len(phrase)
    return [
        d
        for d, doc in enumerate(docs)
        if any(doc[i : i + n] == phrase for i in range(len(doc) - n + 1))
    ]


def bench_positional_index(seed: int = 20261231 + 465) -> dict[str, float]:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(20)]
    match = found = 0
    trials = 40
    for _ in range(trials):
        docs = [
            [rng.choice(vocab) for _ in range(rng.randrange(5, 15))]
            for _ in range(rng.randrange(5, 15))
        ]
        idx = build_positional(docs)
        # plant a phrase in a random doc
        phrase = [rng.choice(vocab) for _ in range(3)]
        if docs:
            d0 = rng.randrange(len(docs))
            pos = rng.randrange(len(docs[d0]))
            docs[d0][pos:pos] = phrase
            idx = build_positional(docs)
            found += int(d0 in phrase_docs(idx, phrase))
        match += int(phrase_docs(idx, phrase) == _brute(docs, phrase))
    return {
        "synthetic_phrase_matches_brute": float(match / trials),
        "synthetic_planted_phrase_found": float(found / trials),
    }
