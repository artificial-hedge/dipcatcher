"""df-pn canon: depth-first proof-number search (Nagai) with a
transposition table — O(depth) memory instead of best-first's
O(tree) frontier — on the subtraction-race DAG, compared against
the wave's best-first proof_number_search. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

INF = 1 << 30

State = tuple[int, bool]  # (stones, attacker-to-move)


def race_children(s: int, max_take: int = 3) -> list[int]:
    return [s - m for m in range(1, min(max_take, s) + 1)]


def dfpn(
    root_stones: int,
    max_take: int = 3,
    max_calls: int = 200000,
) -> tuple[bool | None, int]:
    """Depth-first proof-number search.

    Standard MID(n, phi, delta): expand a most-proving child within
    the caller's thresholds until the node's numbers exceed them.
    Thresholds follow Kishimoto & Müller:
      OR:  phi_c = min(phi, pn2 + 1), delta_c = delta - dn_rest
      AND: delta_c = min(delta, dn2 + 1), phi_c = phi - pn_rest
    where *_rest is the contribution of siblings other than the
    chosen child and *2 the runner-up's number.

    Returns (proved, mid_calls): True if root pn=0 (attacker win),
    False if dn=0, None on budget.
    """
    tt: dict[State, tuple[int, int]] = {}
    calls = 0

    def init(st: State) -> tuple[int, int]:
        s, is_or = st
        if s == 0:
            return (INF, 0) if is_or else (0, INF)
        k = len(race_children(s, max_take))
        return (1, k) if is_or else (k, 1)

    def node_numbers(st: State) -> tuple[int, int]:
        if st in tt:
            return tt[st]
        if st[0] == 0 or st not in tt:
            return init(st)
        return tt[st]

    def children_of(st: State) -> list[State]:
        s, is_or = st
        return [(c, not is_or) for c in race_children(s, max_take)]

    def compute(st: State) -> tuple[int, int]:
        ch = children_of(st)
        if not ch:
            return init(st)
        nums = [node_numbers(c) for c in ch]
        if st[1]:  # OR
            p = min(n[0] for n in nums)
            d = min(sum(n[1] for n in nums), INF)
        else:
            p = min(sum(n[0] for n in nums), INF)
            d = min(n[1] for n in nums)
        tt[st] = (p, d)
        return p, d

    def mid(st: State, phi: int, delta: int) -> None:
        nonlocal calls
        calls += 1
        if calls > max_calls or phi <= 0 or delta <= 0:
            return
        p, d = compute(st)
        if p == 0 or d == 0 or p >= phi or d >= delta:
            tt[st] = (p, d)
            return
        ch = children_of(st)
        while True:
            p, d = compute(st)
            if p == 0 or d == 0 or p >= phi or d >= delta:
                tt[st] = (p, d)
                return
            nums = [node_numbers(c) for c in ch]
            if st[1]:  # OR: child = argmin pn
                order = sorted(range(len(ch)), key=lambda i: nums[i][0])
                i1 = order[0]
                pn2 = nums[order[1]][0] if len(order) > 1 else INF
                dn_rest = d - nums[i1][1]
                phi_c = min(phi, pn2 + 1)
                delta_c = delta - dn_rest
            else:  # AND: child = argmin dn
                order = sorted(range(len(ch)), key=lambda i: nums[i][1])
                i1 = order[0]
                dn2 = nums[order[1]][1] if len(order) > 1 else INF
                pn_rest = p - nums[i1][0]
                delta_c = min(delta, dn2 + 1)
                phi_c = phi - pn_rest
            if calls > max_calls:
                return
            mid(ch[i1], phi_c, delta_c)

    root: State = (root_stones, True)
    p, d = init(root)
    mid(root, INF, INF)
    p, d = node_numbers(root)
    if calls > max_calls:
        return None, calls
    if p == 0:
        return True, calls
    if d == 0:
        return False, calls
    return None, calls


def bench_dfpn(seed: int = 20261231) -> dict[str, float]:
    """df-pn proof of the race game vs best-first PN expansion count."""
    from quant_fund.models.proof_number import proof_number_search

    out: dict[str, float] = {}
    proved_pn, exp_pn = proof_number_search(31)
    proved_df, calls_df = dfpn(31)
    proved_df32, calls_df32 = dfpn(32)
    out["synthetic_pn_proved_31"] = float(proved_pn is True)
    out["synthetic_dfpn_proved_31"] = float(proved_df is True)
    out["synthetic_dfpn_disproved_32"] = float(proved_df32 is False)
    out["synthetic_pn_expansions_31"] = float(exp_pn)
    out["synthetic_dfpn_calls_31"] = float(calls_df)
    out["synthetic_dfpn_calls_32"] = float(calls_df32)
    return out
