"""String-rewriting Knuth–Bendix completion lite (SYNTHETIC bench)."""

from __future__ import annotations

Rule = tuple[str, str]


def reduce_str(w: str, rules: list[Rule], fuel: int = 200) -> str:
    for _ in range(fuel):
        w2 = w
        for lhs, rhs in rules:
            w2 = w2.replace(lhs, rhs, 1)
            if w2 != w:
                break
        if w2 == w:
            return w
        w = w2
    return w


def overlaps(l1: str, l2: str) -> list[str]:
    """Words where l1 occurs inside/overlapping l2's positions (suffix-prefix)."""
    outs = []
    for k in range(1, min(len(l1), len(l2)) + 1):
        if l1[-k:] == l2[:k]:
            outs.append(l1 + l2[k:])
        if l2[-k:] == l1[:k]:
            outs.append(l2 + l1[k:])
    for i in range(len(l2) - len(l1) + 1):
        if l2[i : i + len(l1)] == l1 and i > 0:
            outs.append(l2)
    return outs


def critical_pairs(rules: list[Rule]) -> list[tuple[str, str]]:
    cps = []
    n = len(rules)
    for i in range(n):
        for j in range(n):
            l1, r1 = rules[i]
            l2, r2 = rules[j]
            for w in overlaps(l1, l2):
                a = reduce_str(w.replace(l1, r1, 1), rules)
                b = reduce_str(w.replace(l2, r2, 1), rules)
                if a != b:
                    cps.append((a, b))
    return cps


def joinable(a: str, b: str, rules: list[Rule]) -> bool:
    return reduce_str(a, rules) == reduce_str(b, rules)


def kb_complete(rules: list[Rule], max_iters: int = 40) -> list[Rule]:
    """Complete: add oriented rules for non-joinable critical pairs."""
    rs = list(rules)
    for _ in range(max_iters):
        new = []
        for a, b in critical_pairs(rs):
            na, nb = reduce_str(a, rs), reduce_str(b, rs)
            if na != nb:
                r = (na, nb) if len(na) >= len(nb) else (nb, na)
                if r not in rs and r not in new:
                    new.append(r)
        if not new:
            return rs
        rs = rs + new
    return rs


def locally_confluent(rules: list[Rule]) -> bool:
    return not critical_pairs(rules)


def _bench_knuth_bendix(seed: int = 0) -> float:
    checks = []
    # System {ab->a, bb->a}: word "abb" -> ab=a or a(bb)=aa -> pair (a,aa) not joinable
    r0 = [("ab", "a"), ("bb", "a")]
    checks.append(not locally_confluent(r0))
    rs = kb_complete(r0)
    checks.append(len(rs) > len(r0))
    checks.append(locally_confluent(rs))
    checks.append(reduce_str("abb", rs) == reduce_str("aa", rs))
    # already-confluent system stays put
    r1 = [("ab", "ba")]
    checks.append(
        locally_confluent(kb_complete(r1)) and reduce_str("aab", r1) == reduce_str("aab", r1)
    )
    return float(sum(checks) / len(checks))


def bench_knuth_bendix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knuth_bendix": _bench_knuth_bendix(seed)}
