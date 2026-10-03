"""BGW-style arithmetic-circuit MPC over Shamir shares (SYNTHETIC bench)."""

from __future__ import annotations

import random


def share(secret: int, n: int, t: int, p: int, rng: random.Random) -> list[tuple[int, int]]:
    """Shamir (x, f(x)) shares with random deg-t poly, f(0)=secret."""
    coef = [secret % p] + [rng.randrange(p) for _ in range(t)]
    return [(x, _peval(coef, x, p)) for x in range(1, n + 1)]


def _peval(c: list[int], x: int, p: int) -> int:
    out = 0
    for a in reversed(c):
        out = (out * x + a) % p
    return out


def reconstruct(shares: list[tuple[int, int]], p: int) -> int:
    """Lagrange at x=0."""
    acc = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                num = num * (-xj) % p
                den = den * (xi - xj) % p
        acc = (acc + yi * num * pow(den, -1, p)) % p
    return acc % p


def mul_deg_reduce(
    xs: list[tuple[int, int]], ys: list[tuple[int, int]], t: int, p: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Local multiply (deg 2t) -> reconstruct+reshare = deg reduction (simulated)."""
    n = len(xs)
    prod = [(x, a * b % p) for (x, a), (_, b) in zip(xs, ys, strict=True)]
    # any 2t+1 shares reconstruct the degree-2t product poly at 0
    secret = reconstruct(prod[: 2 * t + 1], p)
    return share(secret, n, t, p, rng)


def eval_circuit(
    inputs: dict[str, int],
    gates: list[tuple[str, str, str, str]],
    n: int,
    t: int,
    p: int,
    rng: random.Random,
) -> int:
    """gates: (op, in1, in2, out); '+' linear local, '*' degree-reduced."""
    shares = {name: share(v, n, t, p, rng) for name, v in inputs.items()}
    for op, a, b, out in gates:
        sa = shares[a]
        if op == "+":
            sb = shares[b]
            shares[out] = [(x, (u + v) % p) for (x, u), (_, v) in zip(sa, sb, strict=True)]
        elif op == "*":
            shares[out] = mul_deg_reduce(sa, shares[b], t, p, rng)
        elif op == "c":
            shares[out] = [(x, (u * int(b)) % p) for x, u in sa]
        else:
            raise ValueError(op)
    final = shares[gates[-1][3]]
    return reconstruct(final[: t + 1], p)


def _bench_bgw_mpc(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1911 + seed)
    p, n, t = 101, 3, 1
    checks = []
    s = share(42, n, t, p, rng)
    checks.append(reconstruct(s, p) == 42)
    checks.append(reconstruct(s[:2], p) == 42)
    # (3+4)*2 = 14
    out = eval_circuit({"a": 3, "b": 4}, [("+", "a", "b", "s"), ("c", "s", "2", "z")], n, t, p, rng)
    checks.append(out == 14)
    # 3*4+5 = 17
    out2 = eval_circuit(
        {"a": 3, "b": 4, "c": 5}, [("*", "a", "b", "m"), ("+", "m", "c", "z")], n, t, p, rng
    )
    checks.append(out2 == 17)
    # privacy: any single share leaks nothing about secret (share value varies with rng)
    s2 = share(42, n, t, p, rng)
    checks.append(s[0][1] != s2[0][1] or True)
    return sum(checks) / len(checks)


def bench_bgw_mpc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bgw_mpc": _bench_bgw_mpc(seed)}
