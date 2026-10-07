"""PLONK-ish constraint system: gate polynomial + copy constraints (SYNTHETIC).

Gate check per row: qL*a + qR*b + qM*a*b + qO*c + qC = 0 (mod p).
Copy constraints via permutation argument: wires assigned to cycles of
a permutation sigma; check product identity ∏ (w_i + beta*i + gamma) ==
∏ (w_i + beta*sigma(i) + gamma) over a challenge beta/gamma — the
standard grand-product trick in small.
"""

from __future__ import annotations

_SEED = 20261231 + 1051

P = 97

GateRow = tuple[int, int, int, int, int, int]  # (qL,qR,qM,qO,qC) + wires via table


def gate_ok(q: tuple[int, int, int, int, int], a: int, b: int, c: int, p: int = P) -> bool:
    qL, qR, qM, qO, qC = q
    return (qL * a + qR * b + qM * a * b + qO * c + qC) % p == 0


def check_circuit(
    gates: list[tuple[int, int, int, int, int]],
    wires: list[tuple[int, int, int]],
    p: int = P,
) -> bool:
    return all(gate_ok(q, a, b, c, p) for q, (a, b, c) in zip(gates, wires, strict=True))


def permutation_holds(
    values: list[int], sigma: list[int], beta: int, gamma: int, p: int = P
) -> bool:
    """Grand-product copy check: ∏(v_i + beta*i + gamma) ==
    ∏(v_i + beta*sigma(i) + gamma)  (mod p)."""
    lhs = 1
    rhs = 1
    for i, v in enumerate(values):
        lhs = lhs * (v + beta * i + gamma) % p
        rhs = rhs * (v + beta * sigma[i] + gamma) % p
    return lhs == rhs


def cycles_of(sigma: list[int]) -> list[list[int]]:
    seen: set[int] = set()
    out: list[list[int]] = []
    for i in range(len(sigma)):
        if i in seen:
            continue
        cyc = []
        j = i
        while j not in seen:
            seen.add(j)
            cyc.append(j)
            j = sigma[j]
        out.append(cyc)
    return out


def copy_consistent(values: list[int], sigma: list[int]) -> bool:
    """All positions in each sigma-cycle must carry equal values."""
    return all(len({values[i] for i in cyc}) == 1 for cyc in cycles_of(sigma))


def bench_plonkish_gate(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # add gate: q=(1,1,0,-1,0) on a+b=c: 3+4=7
    checks.append(check_circuit([(1, 1, 0, -1, 0)], [(3, 4, 7)]))
    # mul gate: q=(0,0,1,-1,0): 3*4=12
    checks.append(check_circuit([(0, 0, 1, -1, 0)], [(3, 4, 12)]))
    # bad gate: 3+4 != 8
    checks.append(not check_circuit([(1, 1, 0, -1, 0)], [(3, 4, 8)]))
    # const gate: q=(0,0,0,-1,+5): -c+5 = 0 at c=5
    checks.append(check_circuit([(0, 0, 0, -1, 5)], [(0, 0, 5)]))
    # copy constraints: wires [7,7,3] with sigma swapping 0<->1
    checks.append(permutation_holds([7, 7, 3], [1, 0, 2], beta=5, gamma=9))
    checks.append(copy_consistent([7, 7, 3], [1, 0, 2]))
    # value mismatch inside a cycle fails copy-consistency (and usually grand product)
    checks.append(not copy_consistent([7, 8, 3], [1, 0, 2]))
    return {"synthetic_plonkish_gate": float(sum(checks)) / len(checks)}
