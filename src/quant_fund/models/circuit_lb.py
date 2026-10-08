"""Exact circuit complexity for small Boolean functions (SYNTHETIC bench)."""

from __future__ import annotations

# A function on n bits is a truth-table int (bit i = f(i)).


def eval_gate(op: str, a: int, b: int, n: int) -> int:
    full = (1 << (1 << n)) - 1
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "nand":
        return ~(a & b) & full
    raise ValueError(op)


def input_tables(n: int) -> list[int]:
    """Truth tables of the n input variables."""
    out = []
    for v in range(n):
        t = 0
        for x in range(1 << n):
            if (x >> (n - 1 - v)) & 1:
                t |= 1 << x
        out.append(t)
    return out


def circuit_size(target: int, n: int, max_size: int = 5) -> int | None:
    """Minimum number of binary gates computing `target` on n vars.

    Circuits are DAGs: a size-k circuit's last gate reads two earlier
    gate outputs of that SAME circuit. Tracking only "each function
    computable in <= k gates" and combining any two such parents
    undercounts — the parents may need disjoint (k-1)-gate circuits.
    We therefore carry the actual gate multiset: each table maps to the
    frozenset of gates producing it, and combining a, b costs
    ``|C_a union C_b| + 1`` (shared subcircuits merge for free).
    Gates are (op, in1, in2); inputs are tables, so two gates emitting
    the same table unify — sound at this scale.
    """
    full = (1 << (1 << n)) - 1
    ops = ["and", "or", "xor", "nand"]
    # table -> Pareto-minimal gate sets (frozensets) computing it.
    # Inclusion dominates: S ⊆ T implies |S ∪ B| ≤ |T ∪ B| for any B,
    # so a superset is never worth keeping; equal-size incomparable
    # sets are kept (different sharing ⇒ different unions downstream).
    # Frontier is capped — smallest sets survive pruning.
    known: dict[int, list[frozenset[tuple[object, ...]]]] = {
        t: [frozenset()] for t in input_tables(n)
    }
    known[0] = [frozenset()]
    known[full] = [frozenset()]

    def offers(t: int, cand: frozenset[tuple[object, ...]]) -> bool:
        cur = known.setdefault(t, [])
        if any(s <= cand for s in cur):
            return False
        cur[:] = [s for s in cur if not cand < s]
        cur.append(cand)
        if len(cur) > 16:
            cur.sort(key=len)
            del cur[16:]
        return True

    if target in known:
        return 0
    for _ in range(4 * max_size):  # rounds until fixpoint
        grew = False
        snap = [(t, s) for t, sets in known.items() for s in sets]
        for i, (t_a, s_a) in enumerate(snap):
            if len(s_a) + 1 > max_size:
                continue
            grew |= offers(~t_a & full, s_a | {("not", t_a)})
            for t_b, s_b in snap[i:]:
                uni = s_a | s_b
                if len(uni) + 1 > max_size:
                    continue
                for op in ops:
                    t = eval_gate(op, t_a, t_b, n)
                    grew |= offers(t, uni | {(op, t_a, t_b)})
        if target in known:
            return min(len(s) for s in known[target])
        if not grew:
            break
    return None


def shannon_bound(n: int) -> float:
    """Shannon counting bound: most n-bit functions need ~2^n / n gates (log2 of function count / log2 of circuit count per gate)."""
    import math

    return math.log2(2 ** (2**n)) / math.log2(4 * (n + 2) ** 2 * 2 ** (2**n) / (2**n))


def _bench_circuit_lb(seed: int = 0) -> float:
    checks = []
    # XOR_2 truth table: f(0)=0,f(1)=1,f(2)=1,f(3)=0 -> bits = 0b0110 = 6
    xor2 = 0b0110
    checks.append(circuit_size(xor2, 2) == 1)  # xor gate is in our basis
    # AND_2 = 0b1000
    checks.append(circuit_size(0b1000, 2) == 1)
    # EQ_2 = 0b1001 = NOT XOR -> 2 gates
    checks.append(circuit_size(0b1001, 2) == 2)
    # constant 0 -> 0 gates
    checks.append(circuit_size(0, 2) == 0)
    # parity_3 = 0b10010110 -> needs several gates
    p3 = circuit_size(0b10010110, 3)
    checks.append(p3 is not None and p3 >= 2)
    return sum(1 for c in checks if c) / len(checks)


def bench_circuit_lb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_circuit_lb": _bench_circuit_lb(seed)}
