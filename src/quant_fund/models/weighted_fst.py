"""Weighted finite-state transducer over the tropical semiring (min,+) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 552


class WFST:
    """trans[state][sym] = list of (next_state, output_sym, weight)."""

    def __init__(self) -> None:
        self.trans: dict[int, dict[str, list[tuple[int, str, float]]]] = {}
        self.start = 0
        self.final: dict[int, float] = {}

    def map_weight(self, word: list[str]) -> float:
        """Min total weight over paths consuming word (returns inf if none)."""
        cur = {self.start: 0.0}
        for ch in word:
            nxt: dict[int, float] = {}
            for s, w in cur.items():
                for t, _, wt in self.trans.get(s, {}).get(ch, []):
                    nxt[t] = min(nxt.get(t, np.inf), w + wt)
            cur = nxt
            if not cur:
                return float("inf")
        return min((w + self.final.get(s, np.inf) for s, w in cur.items()), default=float("inf"))

    def transduce(self, word: list[str]) -> tuple[list[str], float] | None:
        cur: dict[int, tuple[list[str], float]] = {self.start: ([], 0.0)}
        for ch in word:
            nxt: dict[int, tuple[list[str], float]] = {}
            for s, (out, w) in cur.items():
                for t, o, wt in self.trans.get(s, {}).get(ch, []):
                    cand = (out + [o], w + wt)
                    if t not in nxt or cand[1] < nxt[t][1]:
                        nxt[t] = cand
            cur = nxt
            if not cur:
                return None
        best = min(
            ((o, w + self.final.get(s, np.inf)) for s, (o, w) in cur.items()), key=lambda z: z[1]
        )
        return best if np.isfinite(best[1]) else None


def _enum_oracle(fst: WFST, word: list[str]) -> float:
    best = float("inf")

    def rec(s: int, i: int, w: float) -> None:
        nonlocal best
        if w >= best:
            return
        if i == len(word):
            best = min(best, w + fst.final.get(s, np.inf))
            return
        for t, _, wt in fst.trans.get(s, {}).get(word[i], []):
            rec(t, i + 1, w + wt)

    rec(fst.start, 0, 0.0)
    return best


def bench_weighted_fst(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 40
    w_ok = t_ok = 0
    for _ in range(n):
        fst = WFST()
        ns = rng.randint(2, 5)
        fst.start = 0
        for s in range(ns):
            for ch in "ab":
                for t in range(ns):
                    if rng.random() < 0.4:
                        fst.trans.setdefault(s, {}).setdefault(ch, []).append(
                            (t, ch.upper() if rng.random() < 0.5 else ch, rng.uniform(0, 3))
                        )
            if rng.random() < 0.4:
                fst.final[s] = rng.uniform(0, 1)
        word = [rng.choice(list("ab")) for _ in range(rng.randint(1, 5))]
        got = fst.map_weight(word)
        ref = _enum_oracle(fst, word)
        w_ok += int((np.isinf(got) and np.isinf(ref)) or np.isclose(got, ref))
        tr = fst.transduce(word)
        if np.isinf(ref):
            t_ok += int(tr is None)
        else:
            t_ok += int(tr is not None and np.isclose(tr[1], ref))
    return {
        "synthetic_min_weight_exact": float(w_ok / n),
        "synthetic_transduce_exact": float(t_ok / n),
    }
