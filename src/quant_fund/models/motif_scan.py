"""SYNTHETIC PSSM motif scanning + consensus seeding.

Position-specific scoring matrix built from planted motif instances;
scanning re-detects planted sites at better rank than shuffled noise.
"""

from __future__ import annotations

import math
import random


def build_pssm(sites: list[str], bg: float = 0.25) -> list[dict[str, float]]:
    """PSSM log-odds from aligned sites (pseudocount 0.5)."""
    k = len(sites[0])
    alpha = "ACGT"
    pssm = []
    for i in range(k):
        cnt = {c: 0.5 for c in alpha}
        for s in sites:
            cnt[s[i]] += 1
        tot = sum(cnt.values())
        pssm.append({c: math.log((cnt[c] / tot) / bg) for c in alpha})
    return pssm


def scan(seq: str, pssm: list[dict[str, float]]) -> list[tuple[int, float]]:
    k = len(pssm)
    out = []
    for i in range(len(seq) - k + 1):
        w = seq[i : i + k]
        if any(c not in pssm[0] for c in w):
            continue
        out.append((i, sum(pssm[j][w[j]] for j in range(k))))
    return out


def _plant(rng: random.Random, motif: str, n: int, L: int) -> tuple[str, list[int]]:
    seq = [rng.choice("ACGT") for _ in range(L)]
    pos = sorted(rng.sample(range(L - len(motif) + 1), n))
    for p in pos:
        mut = list(motif)
        if rng.random() < 0.3:
            mut[rng.randrange(len(mut))] = rng.choice("ACGT")
        seq[p : p + len(motif)] = mut
    return "".join(seq), pos


def bench_motif_scan(seed: int = 20261231 + 515) -> dict[str, float]:
    rng = random.Random(seed)
    motif = "GATAAG"
    seq, pos = _plant(rng, motif, 6, 400)
    sites = [seq[p : p + 6] for p in pos]
    pssm = build_pssm(sites)
    hits = scan(seq, pssm)
    hits.sort(key=lambda x: -x[1])
    top = {i for i, _ in hits[: len(pos)]}
    recall = len(top & set(pos)) / len(pos)
    # shuffled-control score distribution separation
    ctrl = "".join(rng.choice("ACGT") for _ in range(400))
    ctrl_hits = scan(ctrl, pssm)
    sep = (
        max(s for _, s in ctrl_hits) < sorted(s for _, s in hits)[-1]
        if ctrl_hits and hits
        else False
    )
    planted_scores = [s for i, s in hits if i in set(pos)]
    bg_scores = [s for i, s in hits if i not in set(pos)]
    auc = sum(1 for a in planted_scores for b in bg_scores if a > b) / max(
        1, len(planted_scores) * len(bg_scores)
    )
    return {
        "synthetic_topk_recall": recall,
        "synthetic_auc_vs_bg": auc,
        "synthetic_beats_control": float(sep),
    }
