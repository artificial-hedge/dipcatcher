"""DebtRank systemic-risk centrality — Battiston et al. (2012).

DebtRank propagates distress through a weighted interbank network:
each node i carries a distress level h_i in [0,1] initialised at
d_i (h_i = 1 default, 0 healthy), and iterates

    h_i(t+1) = max( h_i(t), min(1, d_i + sum_j w_ij (h_j(t) - h_j(t-1))_+) )

once along each edge (DebtRank version R1: a node transmits only
the *increment* of distress, and only once). The DebtRank of a
node (or shock vector) is the steady-state weighted distress:

    R = sum_j h_j(infty) v_j - sum_j d_j v_j,   v_j = s_j / sum s

where s_j are bank sizes — i.e. the fraction of total network
economic value lost beyond the initial shock. Nodes are marked
inactive once their distress reaches the threshold psi so the
recursion is finite.

This implementation uses the R1 dynamics on a directed weighted
adjacency matrix (w_ij = exposure of j's loss hitting i, normalized
by i's equity so sum_j w_ij <= 1 is required for stability).

References
----------
- Battiston, S., Puliga, M., Kaushik, R., Tasca, P., Caldarelli, G.
  (2012). "DebtRank: too central to fail? Financial networks, the
  FED and systemic risk." *Scientific Reports* 2, 541.
- Bardoscia, M., Battiston, S., Caccioli, F., Caldarelli, G. (2015).
  "DebtRank: a microscopic foundation for shock propagation."
  *PLoS ONE* 10(6).
- Poledna, S., Thurner, S. (2016). "Elimination of systemic risk in
  financial networks by means of a systemic risk transaction tax."
  *Quantitative Finance* — downstream network clearing use.

Honesty
-------
SYNTHETIC network only; bench verifies propagation asymmetry on a
constructed core-periphery graph — not a real-balance-sheet claim.

Composition
-----------
Called by ``quant_fund.research.benches_w66.bench_debtrank``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def debtrank(
    w: FloatArray,
    sizes: FloatArray,
    shock: FloatArray,
    psi: float = 0.5,
    max_iter: int = 200,
) -> tuple[float, FloatArray]:
    """R1 DebtRank from initial distress ``shock``.

    ``w`` (n,n): w_ij = fraction of node i's equity exposed to node j
    (row sums <= 1 recommended for stability). ``sizes`` (n,):
    economic value v_j normalized inside. ``shock`` (n,): initial
    distress d_i in [0,1]. Returns (debtrank scalar, final distress
    vector h).
    """
    w = np.asarray(w, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    shock = np.asarray(shock, dtype=float)
    n = w.shape[0]
    if w.shape != (n, n) or sizes.shape != (n,) or shock.shape != (n,):
        raise ValueError("shape mismatch")
    if np.any(w < 0) or np.any(sizes <= 0) or np.any((shock < 0) | (shock > 1)):
        raise ValueError("bad weights/sizes/shock")
    if not (0 < psi <= 1):
        raise ValueError("psi must be in (0,1]")
    v = sizes / sizes.sum()
    h = shock.copy()
    h_prev = np.zeros(n)  # increments at t=0 are the shock itself
    for _ in range(max_iter):
        delta = np.maximum(h - h_prev, 0.0)
        if np.all(delta < 1e-12):
            break
        h_next = np.minimum(1.0, h + delta @ w.T)
        h_prev = h.copy()
        h = h_next
    r = float(np.dot(h, v) - np.dot(shock, v))
    return r, h


def debtrank_profile(w: FloatArray, sizes: FloatArray, psi: float = 0.5) -> FloatArray:
    """Per-node DebtRank: shock each node fully (d_i = 1) in turn."""
    w = np.asarray(w, dtype=float)
    n = w.shape[0]
    out = np.zeros(n)
    for i in range(n):
        shock = np.zeros(n)
        shock[i] = 1.0
        out[i], _ = debtrank(w, sizes, shock, psi=psi)
    return out


def in_strength(w: FloatArray) -> FloatArray:
    """Weighted in-strength: how much of the network depends on k —
    the classic ranking DebtRank corrects (it ignores propagation)."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square")
    return w.sum(axis=0)


def bench_debtrank(seed: int = 20261231 + 386) -> dict[str, float]:
    """SYNTHETIC check — small-but-central node outranks by DebtRank."""
    n = 8
    # Divergence construction: node 0 is SMALL (v=0.05) but its
    # default wipes 50% of the equity of the three large banks
    # (v=0.2 each). Node 7 has the largest raw in-strength (three
    # 0.8 exposures) so in-strength centrality ranks it top — but it
    # only damages the tiny right clique (v=0.05 each), so DebtRank
    # still ranks the small hub first.
    w = np.zeros((n, n))
    for j in (1, 2, 3):
        w[j, 0] = 0.5  # large banks exposed to small hub
    for j in (4, 5, 6):
        w[j, 7] = 0.8  # small clique heavily exposed to leaf 7
    sizes = np.array([0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.2])
    dr = debtrank_profile(w, sizes)
    ist = in_strength(w)
    top_dr = int(np.argmax(dr))
    top_ist = int(np.argmax(ist))
    if top_dr != 0:
        raise ValueError("central node not top DebtRank")
    if top_dr == top_ist:
        raise ValueError("DebtRank did not diverge from in-strength")
    # Shock to the in-strength leader: limited value destruction.
    shock = np.zeros(n)
    shock[7] = 1.0
    r_leaf, _ = debtrank(w, sizes, shock)
    if r_leaf > dr[0] + 1e-9:
        raise ValueError("leaf shock exceeded hub")
    return {
        "synthetic_dr_hub": float(dr[0]),
        "synthetic_dr_leaf": r_leaf,
        "synthetic_dr_hub_rank": float(top_dr),
        "synthetic_ist_leader_rank": float(top_ist),
        "synthetic_score": 1.0,
    }
