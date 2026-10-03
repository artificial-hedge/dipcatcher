"""Codon adaptation index (wave 284).

CAI = geometric mean of relative synonymous-codon usage. A gene built from
the most-used codons scores ~1; a gene sampled uniformly scores lower.
"""

import numpy as np

_SEED = 20261231 + 793

_SYN: dict[str, list[str]] = {
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "A": ["GCT", "GCC", "GCA", "GCG"],
}


def _weights(usage: dict[str, int]) -> dict[str, float]:
    w = {}
    for _aa, codons in _SYN.items():
        mx = max(usage[c] for c in codons)
        for c in codons:
            w[c] = usage[c] / mx
    return w


def cai(seq_codons: list[str], w: dict[str, float]) -> float:
    vals = [w[c] for c in seq_codons]
    return float(np.exp(np.mean(np.log(vals))))


def bench_codon_usage(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # reference usage strongly biased to first codon of each family
    usage: dict[str, int] = {}
    for _aa, codons in _SYN.items():
        for i, c in enumerate(codons):
            usage[c] = 100 - 15 * i
    w = _weights(usage)
    hi = [fam[0] for fam in _SYN.values() for _ in range(10)]
    lo = [fam[int(rng.randint(0, len(fam)))] for fam in _SYN.values() for _ in range(10)]
    cai_hi, cai_lo = cai(hi, w), cai(lo, w)
    return {
        "synthetic_cai_hi": float(cai_hi > 0.95),
        "synthetic_cai_gap": float(cai_hi > 1.2 * cai_lo),
    }
