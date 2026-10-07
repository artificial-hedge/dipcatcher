"""Byte-pair encoding (Sennrich 2016) (SYNTHETIC).

Learn merges on a synthetic token corpus; measure vocab growth,
compression (tokens/char) and roundtrip fidelity vs char-level.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _bpe_train(docs: list[list[str]], n_merges: int) -> list[tuple[str, str]]:
    """Whitespace-pre-tokenized BPE: merge most-frequent adjacent pair."""
    vocab: Counter = Counter()
    for d in docs:
        for w in d:
            vocab[tuple(w)] += 1
    merges = []
    for _i in range(n_merges):
        pairs: Counter = Counter()
        for w, c in vocab.items():
            for a, b in zip(w, w[1:], strict=False):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        new_vocab: Counter = Counter()
        for w, c in vocab.items():
            nw = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    nw.append(w[i] + w[i + 1])
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            new_vocab[tuple(nw)] += c
        vocab = new_vocab
    return merges


def _apply(word: str, merges: list[tuple[str, str]]) -> list[str]:
    toks = list(word)
    for a, b in merges:
        nw = []
        i = 0
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                nw.append(a + b)
                i += 2
            else:
                nw.append(toks[i])
                i += 1
        toks = nw
    return toks


def _synth_words(seed: int, n_docs: int) -> list[list[str]]:
    """Words built from stems + affixes → learnable merges."""
    rng = np.random.default_rng(seed)
    stems = ["alpha", "beta", "gamma", "delta", "kappa", "sigma", "theta", "omega"]
    affs = ["ing", "ed", "ly", "tion", "able", "ness", "al", "er"]
    docs = []
    for _d in range(n_docs):
        doc = []
        for _w in range(rng.integers(15, 40)):
            w = stems[rng.integers(len(stems))]
            if rng.uniform() < 0.7:
                w += affs[rng.integers(len(affs))]
            doc.append(w)
        docs.append(doc)
    return docs


def bench_tokenizer_bpe(
    seed: int = 587,
    n_docs: int = 120,
    n_merges: int = 60,
) -> dict[str, float]:
    docs = _synth_words(seed, n_docs)
    merges = _bpe_train(docs, n_merges)
    n_chars = sum(len(w) for d in docs for w in d)
    n_toks_bpe = sum(len(_apply(w, merges)) for d in docs for w in d)
    n_toks_char = n_chars
    # held-out roundtrip
    docs2 = _synth_words(seed + 7, 40)
    roundtrip = all("".join(_apply(w, merges)) == w for d in docs2 for w in d)
    return {
        "synthetic_bpe_compression": n_toks_char / max(n_toks_bpe, 1),
        "synthetic_bpe_tokens": float(n_toks_bpe),
        "synthetic_bpe_char_tokens": float(n_toks_char),
        "synthetic_bpe_roundtrip": float(roundtrip),
        "synthetic_bpe_merges": float(len(merges)),
    }
