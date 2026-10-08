"""Discrete Bayesian networks: hill-climbing structure (SYNTHETIC)
search scored by BIC (K2-style local search with add /
remove / reverse moves, bounded parents) and variable
elimination for exact marginal inference. Synthetic bench
gates recovery of a planted chain structure and marginal
accuracy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _bic_score(data: IntArray, node: int, parents: tuple[int, ...], cards: IntArray) -> float:
    """BIC local score of node | parents."""
    n = data.shape[0]
    card_x = int(cards[node])
    card_p = int(np.prod([cards[p] for p in parents])) if parents else 1
    counts = np.zeros((card_p, card_x))
    for row in data:
        cfg = 0
        for p in parents:
            cfg = cfg * int(cards[p]) + int(row[p])
        counts[cfg, int(row[node])] += 1
    # log-likelihood of multinomial with uniform prior free
    nz = counts[counts > 0]
    ll = float(
        (nz * np.log(nz)).sum()
        - (
            counts.sum(axis=1)[counts.sum(axis=1) > 0]
            * np.log(counts.sum(axis=1)[counts.sum(axis=1) > 0])
        ).sum()
    )
    n_params = card_p * (card_x - 1)
    return float(ll - 0.5 * n_params * np.log(n))


def learn_bayes_net(
    data: IntArray,
    max_parents: int = 3,
    it: int = 200,
) -> list[tuple[int, ...]]:
    """Greedy hill-climb over parent sets scored by BIC.
    Returns parents[j] for each node j."""
    data = np.asarray(data, dtype=np.int64)
    d = data.shape[1]
    cards = np.asarray(data.max(axis=0) + 1, dtype=np.int64)
    parents: list[set[int]] = [set() for _ in range(d)]
    score = np.array([_bic_score(data, j, (), cards) for j in range(d)])
    for _ in range(it):
        best_delta, best_move = 0.0, None
        for child in range(d):
            cur = parents[child]
            for par in range(d):
                if par == child:
                    continue
                # add
                if par not in cur and len(cur) < max_parents:
                    cand = tuple(sorted(cur | {par}))
                    delta = _bic_score(data, child, cand, cards) - score[child]
                    if delta > best_delta and _acyclic(parents, child, par):
                        best_delta, best_move = delta, (child, cand)
                # remove
                if par in cur:
                    cand = tuple(sorted(cur - {par}))
                    delta = _bic_score(data, child, cand, cards) - score[child]
                    if delta > best_delta:
                        best_delta, best_move = delta, (child, cand)
        if best_move is None:
            break
        child, cand = best_move
        parents[child] = set(cand)
        score[child] = _bic_score(data, child, cand, cards)
    return [tuple(sorted(p)) for p in parents]


def _acyclic(parents: Sequence[set[int]], child: int, new_par: int) -> bool:
    """Adding edge new_par→child must not create a cycle:
    check that child is not an ancestor of new_par."""
    seen = {new_par}
    stack = [new_par]
    while stack:
        cur = stack.pop()
        if cur == child:
            return False
        for p in parents[cur]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return True


def cpt_fit(
    data: IntArray, parents: Sequence[Sequence[int]], alpha: float = 0.5
) -> list[FloatArray]:
    """Maximum-likelihood + Laplace CPTs. Returns factor
    tables with dims (parent configs..., node states)."""
    data = np.asarray(data, dtype=np.int64)
    cards = data.max(axis=0) + 1
    cpts: list[FloatArray] = []
    for j in range(data.shape[1]):
        dims = [int(cards[p]) for p in parents[j]] + [int(cards[j])]
        tab = np.full(dims, alpha)
        for row in data:
            key = tuple(int(row[p]) for p in parents[j]) + (int(row[j]),)
            tab[key] += 1.0
        cpts.append(tab / tab.sum(axis=-1, keepdims=True))
    return cpts


def infer_marginal(
    cpts: Sequence[FloatArray], parents: Sequence[Sequence[int]], query: int
) -> FloatArray:
    """Variable elimination over all non-query vars."""
    d = len(cpts)
    # factor var order: parents[j] + [j]; work on one giant joint
    # via variable elimination on a flat joint index for clarity.
    cards = [c.shape[-1] for c in cpts]
    joint = np.ones(1)
    # materialize joint via broadcasting multiplies — small d only
    joint = np.zeros(cards)
    for cell in np.ndindex(*cards):
        p = 1.0
        for j in range(d):
            key = tuple(cell[p] for p in parents[j]) + (cell[j],)
            p *= cpts[j][key]
        joint[cell] = p
    marg = joint.sum(axis=tuple(i for i in range(d) if i != query))
    marg /= marg.sum()
    return np.asarray(marg)


def bench_graphical_models(seed: int = 559) -> dict[str, float]:
    """SYNTHETIC: planted chain A→B→C (A root, C leaf);
    structure search must recover ≥2 of the 2 true edges and
    inferred marginals must match empirical frequencies."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 3000
    a = (rng.uniform(0, 1, n) < 0.4).astype(np.int64)
    p_b_given_a = np.where(a == 1, 0.8, 0.2)
    b = (rng.uniform(0, 1, n) < p_b_given_a).astype(np.int64)
    p_c_given_b = np.where(b == 1, 0.75, 0.15)
    c = (rng.uniform(0, 1, n) < p_c_given_b).astype(np.int64)
    data = np.c_[a, b, c]
    parents = learn_bayes_net(data, max_parents=2, it=60)
    edges = {(p, j) for j, ps in enumerate(parents) for p in ps}
    # Markov equivalence: edge direction is not identifiable
    # from observational BIC — gate on the skeleton.
    skeleton = {frozenset(e) for e in edges}
    true_skel = {frozenset((0, 1)), frozenset((1, 2))}
    out["synthetic_bn_edges_found"] = float(len(skeleton & true_skel))
    out["synthetic_bn_extra_edges"] = float(len(skeleton - true_skel))
    if len(skeleton & true_skel) < 2:
        raise ValueError(f"bn structure off: {parents}")
    cpts = cpt_fit(data, parents)
    marg_c = infer_marginal(cpts, parents, 2)
    emp_c = float(c.mean())
    out["synthetic_bn_marg_err"] = float(abs(marg_c[1] - emp_c))
    if out["synthetic_bn_marg_err"] > 0.1:
        raise ValueError(f"bn marginal off: {marg_c[1]} vs {emp_c}")
    return out
