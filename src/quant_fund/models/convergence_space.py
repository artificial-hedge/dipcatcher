"""Sequence convergence on finite topologies; Hausdorff => unique limits (SYNTHETIC)."""

from __future__ import annotations

Topo = frozenset[frozenset[int]]


def seq_converges(seq: list[int], limit: int, opens: Topo) -> bool:
    """x_n -> L iff for every open U containing L, seq eventually in U."""
    return all(not (limit in u and not all(x in u for x in seq)) for u in opens)


def all_limits(seq: list[int], univ: frozenset[int], opens: Topo) -> frozenset[int]:
    return frozenset(lim for lim in univ if seq_converges(seq, lim, opens))


def is_hausdorff(univ: frozenset[int], opens: Topo) -> bool:
    return all(
        x == y or any(x in u and y in v and not (u & v) for u in opens for v in opens)
        for x in univ
        for y in univ
    )


def _bench_convergence_space(seed: int = 0) -> float:
    checks = []
    u = frozenset({0, 1})
    ind = frozenset({frozenset(), u})
    disc = frozenset({frozenset(), frozenset({0}), frozenset({1}), u})
    seq = [0, 0, 0, 0]
    checks.append(all_limits(seq, u, ind) == frozenset({0, 1}))  # indiscrete: conv to all
    checks.append(all_limits(seq, u, disc) == frozenset({0}))  # discrete: only to 0
    checks.append(is_hausdorff(u, disc))
    checks.append(not is_hausdorff(u, ind))
    # Hausdorff iff constant sequence has unique limit
    checks.append(len(all_limits(seq, u, disc)) == 1)
    checks.append(len(all_limits(seq, u, ind)) == 2)
    return float(sum(checks) / len(checks))


def bench_convergence_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_convergence_space": _bench_convergence_space(seed)}
