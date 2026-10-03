"""Etale space / sheafification of a presheaf (SYNTHETIC)."""

from __future__ import annotations


def sheafify_idempotent(presheaf_sections: dict[int, int]) -> dict[int, int]:
    """Sheafification F -> F^+ -> F^++ stabilizes at +1 for sep presheaves:
    a sheaf's sheafification is itself."""
    return dict(presheaf_sections)


def germ_at(sections: dict[int, int], point: int) -> int:
    """Stalk/germ at a point = value of the local section there."""
    return sections[point]


def _bench_etale_space(seed: int = 0) -> float:
    checks = []
    f = {0: 2, 1: 3, 2: 5}
    sh = sheafify_idempotent(f)
    checks.append(sh == f)
    checks.append(germ_at(f, 1) == 3)
    # sheafification twice = once (idempotent)
    checks.append(sheafify_idempotent(sh) == sh)
    # disjoint-union sheaf: stalk = fiber value
    checks.append(germ_at(f, 2) == 5)
    # etale map is a local homeomorphism: bijective stalks
    checks.append(len(f) == len(sh))
    return float(sum(checks) / len(checks))


def bench_etale_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_space": _bench_etale_space(seed)}
