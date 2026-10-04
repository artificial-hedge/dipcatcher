"""Hall pi-subgroups of solvable groups (SYNTHETIC)."""

from __future__ import annotations


def is_hall_subgroup(order_g: int, order_h: int, pi: set[int]) -> bool:
    """H is a Hall pi-subgroup: |H| is a pi-number and gcd(|H|, [G:H]) = 1."""
    from math import gcd

    if order_g % order_h:
        return False
    idx = order_g // order_h
    if gcd(order_h, idx) != 1:
        return False
    # every prime divisor of |H| must lie in pi
    d = order_h
    p = 2
    while d > 1:
        if d % p == 0:
            if p not in pi:
                return False
            d //= p
        else:
            p += 1
    return True


def _bench_hall_subgroup(seed: int = 0) -> float:
    checks = []
    # A4 (order 12): V4 (order 4) is a Hall {2}-subgroup (a Sylow)
    checks.append(is_hall_subgroup(12, 4, {2}))
    # C3 (order 3) is a Hall {3}-subgroup
    checks.append(is_hall_subgroup(12, 3, {3}))
    # the whole group is a Hall {2,3}-subgroup
    checks.append(is_hall_subgroup(12, 12, {2, 3}))
    # C2 (order 2) is NOT a Hall {2}-subgroup: index 6 shares factor 2
    checks.append(not is_hall_subgroup(12, 2, {2}))
    # order-6 subgroup in S3-ish group: S3 (order 6): C3 is Hall{3}
    checks.append(is_hall_subgroup(6, 3, {3}))
    checks.append(is_hall_subgroup(6, 2, {2}))
    # order-4 in order-8 group is not Hall (index 2 shares factor)
    checks.append(not is_hall_subgroup(8, 4, {2}))
    # order-4 IS Hall{2} in order-12 only if index 3 coprime: yes above
    return float(sum(checks) / len(checks))


def bench_hall_subgroup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hall_subgroup": _bench_hall_subgroup(seed)}
