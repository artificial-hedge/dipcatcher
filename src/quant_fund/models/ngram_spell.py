"""SYNTHETIC n-gram spell checker.

Bigram-overlap candidate generation + edit-distance rerank; correct
suggestion = first ranked candidate equals ground-truth word.
"""

from __future__ import annotations

import random


def _bigrams(w: str) -> set[str]:
    w = f"^{w}$"
    return {w[i : i + 2] for i in range(len(w) - 1)}


def _ed(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def suggest(word: str, vocab: list[str], grams: dict[str, set[int]]) -> str | None:
    g = _bigrams(word)
    cand: dict[int, int] = {}
    for x in g:
        for i in grams.get(x, ()):  # overlap count
            cand[i] = cand.get(i, 0) + 1
    if not cand:
        return None
    top = sorted(cand, key=lambda i: (-cand[i], i))[:10]
    if not top:
        return None
    best = min(top, key=lambda i: (_ed(word, vocab[i]), i))
    return vocab[best]


def _mutate(rng: random.Random, w: str) -> str:
    chars = list(w)
    op = rng.randrange(4)
    i = rng.randrange(len(chars))
    if op == 0:
        chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    elif op == 1 and len(chars) > 3:
        del chars[i]
    elif op == 2:
        chars.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif op == 3 and i + 1 < len(chars):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def bench_ngram_spell(seed: int = 20261231 + 464) -> dict[str, float]:
    rng = random.Random(seed)
    words = [
        "portfolio",
        "volatility",
        "regression",
        "momentum",
        "arbitrage",
        "covariance",
        "liquidity",
        "benchmark",
        "execution",
        "sentiment",
        "derivative",
        "hedging",
        "drawdown",
        "kurtosis",
        "stationary",
        "cointegration",
        "leverage",
        "dividend",
        "treasury",
        "inflation",
    ]
    grams: dict[str, set[int]] = {}
    for i, w in enumerate(words):
        for g in _bigrams(w):
            grams.setdefault(g, set()).add(i)
    hit = has_cand = 0
    trials = 60
    for _ in range(trials):
        w = rng.choice(words)
        miss = _mutate(rng, w)
        s = suggest(miss, words, grams)
        has_cand += int(s is not None)
        hit += int(s == w)
    return {
        "synthetic_top1_accuracy": float(hit / trials),
        "synthetic_candidate_coverage": float(has_cand / trials),
    }
