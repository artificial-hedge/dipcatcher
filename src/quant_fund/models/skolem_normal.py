"""Skolemization: forall-exists -> function symbols, satisfiability kept (SYNTHETIC)."""

from __future__ import annotations


def _bench_skolem_normal(seed: int = 0) -> float:
    checks = []
    # theory: forall x exists y. y > x (over {0,1,2} bounded model -> fails at top)
    domain = [0, 1, 2]
    holds_forall_exists = all(any(y > x for y in domain) for x in domain)
    checks.append(not holds_forall_exists)  # fails at x=2

    # skolemized: exists f. forall x. f(x) > x -- same model-theoretic failure
    def skolem_ok(dom: list[int]) -> bool:
        for seed_fn in range(len(dom) ** len(dom)):
            f = [0] * len(dom)
            tmp = seed_fn
            for i in range(len(dom)):
                f[i] = tmp % len(dom)
                tmp //= len(dom)
            if all(f[x] > dom[x] for x in range(len(dom))):
                return True
        return False

    checks.append(not skolem_ok(domain))
    # satisfiable example: forall x exists y. y = x -> holds; Skolem f = id
    checks.append(skolem_ok(domain) or all(any(y == x for y in domain) for x in domain))
    checks.append(all(any(y == x for y in domain) for x in domain))
    # forall-exists alternation: forall x exists y. x != y on |dom| >= 2
    checks.append(all(any(y != x for y in domain) for x in domain))
    # equivalent skolem: forall x. x != f(x) with f = successor mod n works
    n = len(domain)
    f = [(x + 1) % n for x in domain]
    checks.append(all(domain[x] != f[x] for x in range(n)))
    return float(sum(checks) / len(checks))


def bench_skolem_normal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skolem_normal": _bench_skolem_normal(seed)}
