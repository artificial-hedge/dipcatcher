"""Profile HMM: consensus detection over a multiple alignment."""

import numpy as np

_SEED = 20261231 + 728
_ALPH = "ACGT"


def build_profile(alignment: list[str], pseudocount: float = 0.1) -> np.ndarray:
    """Column-wise emission probs (L × 4) with pseudocounts."""
    L = len(alignment[0])
    prof = np.full((L, 4), pseudocount)
    for seq in alignment:
        for j, c in enumerate(seq):
            prof[j, _ALPH.index(c)] += 1.0
    prof /= prof.sum(axis=1, keepdims=True)
    return prof


def loglik(seq: str, prof: np.ndarray) -> float:
    return float(np.log([prof[j, _ALPH.index(c)] for j, c in enumerate(seq)]).sum())


def bench_hmm_profile(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    L, n = 10, 20
    # planted motif: position j prefers base j % 4
    cons = np.array([j % 4 for j in range(L)])
    alignment = []
    for _ in range(n):
        s = "".join(_ALPH[c] if rng.rand() < 0.8 else _ALPH[rng.randint(4)] for c in cons)
        alignment.append(s)
    prof = build_profile(alignment)
    inlike = np.array([loglik(s, prof) for s in alignment])
    outs = np.array(
        [loglik("".join(_ALPH[rng.randint(4)] for _ in range(L)), prof) for _ in range(20)]
    )
    sep = float(inlike.mean() - outs.mean())
    auc = float((inlike[:, None] > outs[None, :]).mean())
    return {"synthetic_profile_sep": float(sep > 0), "synthetic_profile_auc": auc}
