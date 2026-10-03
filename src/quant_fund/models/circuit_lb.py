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
        return (~a & full) if True else 0
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

    BFS over sets of computable tables: level k = functions computable with
    k gates (NOT folded as 1-input gate).
    """
    full = (1 << (1 << n)) - 1
    known = set(input_tables(n))
    known |= {0, full}
    if target in known:
        return 0
    ops = ["and", "or", "xor", "nand"]
    for size in range(1, max_size + 1):
        new: set[int] = set()
        prev = list(known)
        for a in prev:
            na = ~a & full
            if na not in known and na not in new:
                new.add(na)
        for i, a in enumerate(prev):
            for b in prev[i:]:
                for op in ops:
                    t = eval_gate(op, a, b, n)
                    if t not in known and t not in new:
                        new.add(t)
        if target in new:
            return size
        known |= new
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
