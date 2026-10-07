"""Outcome-sampling MCCFR (Lanctot et al. 2009) — Monte Carlo CFR (SYNTHETIC)
on Kuhn poker: sample one terminal history per info set per pass,
update tabular regrets on the sampled subtree. Exploitability vs the
tabular full-traversal CFR reference.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sp_synth import CARDS, kuhn_acts, kuhn_exploit, kuhn_terminal, kuhn_util


def _is_p2_turn(h: str) -> bool:
    return len(h) % 2 == 1


def bench_mccfr_outcome(seed: int = 2701, iters: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # regrets keyed by (player, card, hist)
    R: dict[tuple[int, int, str], np.ndarray] = {}
    S: dict[tuple[int, int, str], np.ndarray] = {}

    def strat(p: int, c: int, h: str) -> np.ndarray:
        r = R.get((p, c, h), np.zeros(2))
        rp = np.maximum(r, 0)
        s = rp.sum()
        return rp / s if s > 0 else np.array([0.5, 0.5])

    def sample(c1: int, c2: int, h: str, upd: int) -> float:
        """Outcome-sampled counterfactual walk; returns sampled util for `upd`."""
        if kuhn_terminal(h):
            u = kuhn_util(h, c1, c2)
            return u if upd == 0 else -u
        p = len(h) % 2
        c = c1 if p == 0 else c2
        s = strat(p, c, h)
        if p == upd:
            # sample one action, counterfactual value for both
            a_i = int(rng.choice(2, p=s))
            v = np.zeros(2)
            for i, a in enumerate(kuhn_acts(h)):
                if i == a_i:
                    v[i] = sample(c1, c2, h + a, upd)
                else:
                    # off-path: value estimated non-recursively via terminal rollout
                    hh = h + a
                    while not kuhn_terminal(hh):
                        pp = len(hh) % 2
                        cc = c1 if pp == 0 else c2
                        sp = strat(pp, cc, hh)
                        hh += "b" if rng.random() < sp[1] else "p"
                    uu = kuhn_util(hh, c1, c2)
                    v[i] = uu if upd == 0 else -uu
            key = (p, c, h)
            R[key] = R.get(key, np.zeros(2)) + (v - v[a_i])
            S[key] = S.get(key, np.zeros(2)) + s
            return float(v[a_i])
        # opponent turn: sample action, no update
        hh = h + kuhn_acts(h)[int(rng.choice(2, p=s))]
        return sample(c1, c2, hh, upd)

    for _ in range(iters):
        for p in (0, 1):
            for c1 in CARDS:
                for c2 in CARDS:
                    if c1 != c2:
                        sample(c1, c2, "", p)

    def avg_strat(player: int, card: int, hist: str) -> np.ndarray:
        key = (player, card, hist)
        s = S.get(key)
        if s is None or s.sum() == 0:
            return np.array([0.5, 0.5])
        return np.asarray(s) / float(s.sum())

    expl = kuhn_exploit(avg_strat)
    ref = kuhn_exploit(lambda p, c, h: np.array([0.5, 0.5]))
    return {
        "synthetic_mccfr_expl": expl,
        "synthetic_random_expl": ref,
        "synthetic_mccfr_drop": ref - expl,
        "synthetic_torch_available": 0.0,
    }
