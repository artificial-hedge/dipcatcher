"""Deep CFR / strategy amortization (Brown et al. 2019 adapted) — (SYNTHETIC)
tabular outcome CFR generates counterfactual-value + average-strategy
targets on Kuhn; policy nets are trained to amortize both. Reports the
exploitability of the net policy vs its tabular teacher — the
function-approximation gap (always positive, small if amortization
succeeds).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sp_synth import (
    CARDS,
    encode_kuhn,
    kuhn_acts,
    kuhn_exploit,
    kuhn_terminal,
    kuhn_util,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deep_cfr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deep_cfr(seed: int = 2707, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    R: dict[tuple[int, int, str], np.ndarray] = {}
    S: dict[tuple[int, int, str], np.ndarray] = {}

    def strat_t(p: int, c: int, h: str) -> np.ndarray:
        rp = np.maximum(R.get((p, c, h), np.zeros(2)), 0)
        return rp / rp.sum() if rp.sum() > 0 else np.array([0.5, 0.5])

    def cfr(c1: int, c2: int, h: str, r0: float, r1: float) -> float:
        if kuhn_terminal(h):
            return kuhn_util(h, c1, c2)
        p = len(h) % 2
        c = c1 if p == 0 else c2
        s = strat_t(p, c, h)
        v = np.zeros(2)
        for i, a in enumerate(kuhn_acts(h)):
            nr0, nr1 = r0, r1
            if p == 0:
                nr0 *= s[i]
            else:
                nr1 *= s[i]
            v[i] = cfr(c1, c2, h + a, nr0, nr1)
        opp = r1 if p == 0 else r0
        vp = v if p == 0 else -v  # counterfactual values in player p's utility
        R[(p, c, h)] = R.get((p, c, h), np.zeros(2)) + opp * (vp - vp @ s)
        own = r0 if p == 0 else r1
        S[(p, c, h)] = S.get((p, c, h), np.zeros(2)) + own * s
        return float(v @ s)

    for _ in range(iters):
        for c1 in CARDS:
            for c2 in CARDS:
                if c1 != c2:
                    cfr(c1, c2, "", 1.0, 1.0)

    def tab_avg(p: int, c: int, h: str) -> np.ndarray:
        s = S.get((p, c, h))
        if s is None or s.sum() == 0:
            return np.array([0.5, 0.5])
        return np.asarray(s) / float(s.sum())

    teacher_expl = kuhn_exploit(tab_avg)

    # amortize: net learns the tabular average strategy over infosets
    infos = [
        (p, c, h)
        for p in (0, 1)
        for c in CARDS
        for h in ("", "p", "b", "pb")
        if len(h) % 2 == p and not kuhn_terminal(h)
    ]
    X = torch.tensor(np.array([encode_kuhn(p, c, h) for p, c, h in infos])).float()
    Y = torch.tensor(np.array([tab_avg(p, c, h) for p, c, h in infos])).float()
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(300):
        loss = ((net(X).softmax(-1) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def net_strat(p: int, c: int, h: str) -> np.ndarray:
        with torch.no_grad():
            out = net(torch.tensor(encode_kuhn(p, c, h)).float()).softmax(-1).numpy()
            return np.asarray(out, dtype=np.float64)

    net_expl = kuhn_exploit(net_strat)
    return {
        "synthetic_dcfr_expl": net_expl,
        "synthetic_tabular_expl": teacher_expl,
        "synthetic_amortize_gap": net_expl - teacher_expl,
        "synthetic_torch_available": 1.0,
    }
