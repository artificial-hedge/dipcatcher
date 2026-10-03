"""Say-Echo-Do narrative sentiment engine (Exec-Summary NLP item).
Institutions issue statements ("Say"), media articles echo them
("Echo"), and institutions' actual positioning ("Do") covaries with
future returns — contrarian when say and do disagree.

Pipeline: deterministic hashing vectorizer -> return-aligned contrastive
embedding -> echo detection vs recent statements -> per-institution
say-do covariance -> contrarian signal.

Synthetic bench: echo-detection AUC, say-do covariance sign, contrarian
signal accuracy on next-day returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray

_POS = {"beat", "surge", "upgrade", "growth", "bullish", "strong", "record"}
_NEG = {"miss", "drop", "downgrade", "weak", "bearish", "risk", "slump"}


def vectorize(text: str, dim: int = 64) -> FloatArray:
    """Deterministic hashing bag-of-words -> L2-normalized dense vector."""
    v = np.zeros(dim)
    for tok in text.lower().split():
        h = hash(tok) % dim
        v[h] += 1.0 if tok not in _NEG else -1.0
        if tok in _POS:
            v[h] += 0.5
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def tone(text: str) -> float:
    toks = text.lower().split()
    pos = sum(t in _POS for t in toks)
    neg = sum(t in _NEG for t in toks)
    return (pos - neg) / max(pos + neg, 1)


@dataclass
class ContrastiveEmbedder:
    """Return-aligned linear probe over hashed embeddings."""

    dim: int = 64

    def __post_init__(self) -> None:
        self.w = np.zeros(self.dim)

    def fit(
        self,
        embs: FloatArray,
        fwd_ret_sign: FloatArray,
        lr: float = 0.5,
        iters: int = 300,
    ) -> None:
        y = np.sign(fwd_ret_sign)
        for _ in range(iters):
            z = np.clip(embs @ self.w, -20, 20)
            p = 1.0 / (1.0 + np.exp(-z))
            self.w += lr * (embs.T @ (0.5 * (y + 1) - p)) / len(y)

    def score(self, emb: FloatArray) -> float:
        return float(emb @ self.w)


def echo_score(article_emb: FloatArray, stmt_embs: FloatArray) -> float:
    """Max cosine similarity of article to recent institution statements."""
    if len(stmt_embs) == 0:
        return 0.0
    a = article_emb / max(np.linalg.norm(article_emb), 1e-9)
    s = stmt_embs / np.linalg.norm(stmt_embs, axis=1, keepdims=True) + 1e-9
    return float(np.max(s @ a))


@dataclass
class SayDoTracker:
    """Rolling say-tone vs do-flow covariance per institution."""

    window: int = 30

    def __post_init__(self) -> None:
        self.says: dict[int, list[float]] = {}
        self.dos: dict[int, list[float]] = {}

    def update(self, inst: int, say: float, do: float) -> None:
        self.says.setdefault(inst, []).append(say)
        self.dos.setdefault(inst, []).append(do)
        for d in (self.says[inst], self.dos[inst]):
            del d[: max(0, len(d) - self.window)]

    def covariance(self, inst: int) -> float:
        s = np.asarray(self.says.get(inst, []))
        d = np.asarray(self.dos.get(inst, []))
        if len(s) < 5 or s.std() < 1e-9 or d.std() < 1e-9:
            return 0.0
        return float(np.cov(s, d)[0, 1] / (s.std() * d.std()))


def synth_market(days: int, rng: np.random.Generator) -> dict:
    """One institution that speaks bearish while accumulating.

    do>0 & say<0 -> forward returns positive (contrarian doc finding).
    """
    inst_say = -np.abs(rng.normal(0.5, 0.2, days))
    hidden_buy = rng.random(days) < 0.5
    inst_do = np.where(hidden_buy, rng.uniform(0.4, 1.0, days), rng.uniform(-0.6, 0.2, days))
    fwd_ret = np.zeros(days)
    for t in range(days):
        fwd_ret[t] = (
            0.01 * np.sign(inst_do[t]) - 0.004 * inst_say[t] + 0.015 * rng.standard_normal()
        )
    # articles: echoes share the institution's negative tone
    stmts = [vectorize("bearish risk weak downgrade slump")]
    arts, is_echo = [], np.zeros(days)
    for t in range(days):
        if rng.random() < 0.7:
            arts.append(vectorize(f"institution warns bearish risk weak slump day{t}"))
            is_echo[t] = 1.0
        else:
            arts.append(vectorize(f"analyst sees growth surge bullish strong day{t}"))
    return {
        "say": inst_say,
        "do": inst_do,
        "fwd": fwd_ret,
        "stmts": np.array(stmts),
        "arts": np.array(arts),
        "is_echo": is_echo,
    }


def bench_say_echo_do(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    m = synth_market(400, rng)
    trk = SayDoTracker()
    covs, contrarian_hits = [], []
    for t in range(50, len(m["say"]) - 1):
        trk.update(0, m["say"][t], m["do"][t])
        cov = trk.covariance(0)
        covs.append(cov)
        # contrarian signal: say<0 & do>0 -> buy
        sig = m["say"][t] < 0 and m["do"][t] > 0
        contrarian_hits.append(float(np.sign(m["fwd"][t + 1]) > 0) if sig else np.nan)
    hits = [h for h in contrarian_hits if not np.isnan(h)]
    # echo detection AUC
    scores = np.array([echo_score(a, m["stmts"]) for a in m["arts"]])
    y = m["is_echo"]
    order = np.argsort(scores)
    ranks = np.empty(len(y))
    ranks[order] = np.arange(1, len(y) + 1)
    pos = y == 1
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    # return-aligned probe on article embeddings
    emb = ContrastiveEmbedder()
    emb.fit(m["arts"][:200], m["fwd"][:200])
    probe_acc = float(
        np.mean(np.sign([emb.score(a) for a in m["arts"][200:]]) == np.sign(m["fwd"][200:]))
    )
    return {
        "synthetic_sed_echo_auc": auc,
        "synthetic_sed_mean_say_do_corr": float(np.mean(covs)),
        "synthetic_sed_contrarian_accuracy": float(np.mean(hits)) if hits else 0.0,
        "synthetic_sed_contrarian_coverage": float(len(hits) / len(contrarian_hits)),
        "synthetic_sed_probe_accuracy": probe_acc,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_say_echo_do(), indent=1))
