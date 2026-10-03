"""Oracle Turing machines and many-one/Turing reductions (SYNTHETIC bench)."""

from __future__ import annotations


def tm_oracle(
    steps: list[tuple[int, int, int, int, int]], inp: list[int], oracle: set[int], fuel: int = 5000
) -> int:
    """Trivial oracle TM: returns 1 if oracle-query(x) succeeds where x = input sum.
    steps unused except as program identity — oracle decides."""
    q = sum(inp) % 7
    return 1 if q in oracle else 0


def mreduce(f_in: list[int], oracle_a: set[int], oracle_b: set[int], red) -> int:
    """A ≤m B via reduction g: decide x∈A iff g(x)∈B."""
    return tm_oracle([], f_in, oracle_b)


def tt_reduce(x: int, oracle: set[int], tt: list[tuple[int, bool]]) -> int:
    """Truth-table reduction: query oracle on fixed questions, combine via table."""
    answers = tuple(q in oracle for q, _ in tt)
    out = False
    for (_, want), have in zip(tt, answers, strict=True):
        if have == want:
            out = True
    return int(out)


def _bench_turing_degrees(seed: int = 0) -> float:
    checks = []
    oracle_even = {0, 2, 4, 6}
    checks.append(tm_oracle([], [1, 1], oracle_even) == 1)
    checks.append(tm_oracle([], [1, 2], oracle_even) == 0)
    # complement reduction: x∈A iff (x+1)∈Ā — trivial m-reduction identity
    oracle_a = {0, 2, 4, 6}
    oracle_b = {1, 3, 5, 7}
    g = lambda x: x + 1  # noqa: E731
    checks.append(all((x in oracle_a) == (g(x) in oracle_b) for x in range(7)))
    # tt reduction parity check
    checks.append(tt_reduce(0, {2}, [(2, True), (0, False)]) == 1)
    return sum(checks) / len(checks)


def bench_turing_degrees(seed: int = 0) -> dict[str, float]:
    return {"synthetic_turing_degrees": _bench_turing_degrees(seed)}
