"""AND-inverter graph (AIG) structural hashing + dead-cone elimination.

Nodes are ANDs with complemented edges (lit = 2*node + phase). Strashing
deduplicates literal-identical fanin pairs; the sweep drops cones not
reachable from outputs. Verified: exact truth-table preservation and a
strict size reduction on a netlist with injected redundancy.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 953


def _lit(node: int, phase: int = 0) -> int:
    return 2 * node + phase


def build_netlist() -> tuple[list[tuple[int, int]], list[int], int]:
    """4-bit xor-tree over 4 inputs plus a deliberately duplicated cone.

    Returns (ands, outputs, n_in). Literals index nodes 0..n_in-1 for PIs.
    """
    n_in = 4
    ands: list[tuple[int, int]] = []

    def a(l1: int, l2: int) -> int:
        ands.append((l1, l2))
        return _lit(n_in + len(ands) - 1)

    def xor(l1: int, l2: int) -> int:
        t1 = a(l1, l2 ^ 1)
        t2 = a(l1 ^ 1, l2)
        return a(t1 ^ 1, t2 ^ 1) ^ 1  # NAND->OR involution

    # f0 = x0^x1, f1 = x0^x1^x2, f2 = x0^x1^x2^x3 (shared cones at build time)
    f0 = xor(_lit(0), _lit(1))
    f1 = xor(f0, _lit(2))
    f2 = xor(f1, _lit(3))
    # redundant duplicate of f0 appended raw (unstrashed copy)
    t1 = a(_lit(0), _lit(1) ^ 1)
    t2 = a(_lit(0) ^ 1, _lit(1))
    f0_dup = a(t1 ^ 1, t2 ^ 1) ^ 1
    outs = [f0, f1, f2, f0_dup]
    return ands, outs, n_in


def eval_aig(ands: list[tuple[int, int]], outs: list[int], n_in: int, x: int) -> int:
    vals = [(x >> i) & 1 for i in range(n_in)]
    for l1, l2 in ands:
        vals.append((vals[l1 >> 1] ^ (l1 & 1)) & (vals[l2 >> 1] ^ (l2 & 1)))
    out = 0
    for k, lit in enumerate(outs):
        out |= (vals[lit >> 1] ^ (lit & 1)) << k
    return out


def strash(
    ands: list[tuple[int, int]], outs: list[int], n_in: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Forward dedup pass: identical literal pairs collapse to one node."""
    seen: dict[tuple[int, int], int] = {}
    remap = list(range(n_in))
    new_ands: list[tuple[int, int]] = []
    for l1, l2 in ands:
        l1 = 2 * remap[l1 >> 1] + (l1 & 1)
        l2 = 2 * remap[l2 >> 1] + (l2 & 1)
        key = (min(l1, l2), max(l1, l2))
        node_new = n_in + len(new_ands)
        if key in seen:
            remap.append(seen[key])
        else:
            seen[key] = node_new
            remap.append(node_new)
            new_ands.append((l1, l2))
    return new_ands, [2 * remap[lit >> 1] + (lit & 1) for lit in outs]


def sweep(
    ands: list[tuple[int, int]], outs: list[int], n_in: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Dead-cone elimination: keep only ANDs reachable from outputs."""
    used = [False] * (n_in + len(ands))
    for lit in outs:
        used[lit >> 1] = True
    for i in range(len(ands) - 1, -1, -1):
        if used[n_in + i]:
            used[ands[i][0] >> 1] = True
            used[ands[i][1] >> 1] = True
    remap = [-1] * (n_in + len(ands))
    for i in range(n_in):
        remap[i] = i
    new_ands: list[tuple[int, int]] = []
    for i, (l1, l2) in enumerate(ands):
        if used[n_in + i]:
            remap[n_in + i] = n_in + len(new_ands)
            new_ands.append((l1, l2))
    # rewire fanins through kept-node remapping
    fix = [(2 * remap[l1 >> 1] + (l1 & 1), 2 * remap[l2 >> 1] + (l2 & 1)) for l1, l2 in new_ands]
    new_outs = [2 * remap[lit >> 1] + (lit & 1) for lit in outs]
    return fix, new_outs


def bench_aig_rewrite(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    ands, outs, n_in = build_netlist()
    ref = [eval_aig(ands, outs, n_in, x) for x in range(1 << n_in)]
    s_ands, s_outs = strash(ands, outs, n_in)
    r_ands, r_outs = sweep(s_ands, s_outs, n_in)
    got = [eval_aig(r_ands, r_outs, n_in, x) for x in range(1 << n_in)]
    checks = [
        len(s_ands) < len(ands),  # duplicate cone collapsed
        len(r_ands) <= len(s_ands),
        got == ref,
        eval_aig(ands, outs, n_in, 0b0101) == 0b1001,  # f0=1,f1=0,f2=0,dup=1
    ]
    return {"synthetic_aig_rewrite": float(np.mean(checks))}
