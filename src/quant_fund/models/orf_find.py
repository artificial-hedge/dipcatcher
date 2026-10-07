"""ORF finder: ATG..stop in all 3 frames (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 731
_STOP = {"TAA", "TAG", "TGA"}


def longest_orf(seq: str) -> int:
    """Longest nt run between ATG and in-frame stop."""
    best = 0
    for frame in range(3):
        i = frame
        while i + 3 <= len(seq):
            codon = seq[i : i + 3]
            if codon == "ATG":
                j = i + 3
                while j + 3 <= len(seq) and seq[j : j + 3] not in _STOP:
                    j += 3
                if j + 3 <= len(seq):
                    best = max(best, j + 3 - i)
                i = j
            i += 3
    return best


def bench_orf_find(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # planted ORF of 15 codons
    planted = "ATG" + "".join("AAA" for _ in range(13)) + "TAA"
    flank = "".join("ACGT"[rng.randint(4)] for _ in range(60))
    seq = flank[:20] + planted + flank[20:]
    found = longest_orf(seq)
    return {"synthetic_orf_exact": float(found >= len(planted))}
