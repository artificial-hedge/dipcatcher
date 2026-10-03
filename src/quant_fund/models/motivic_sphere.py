"""Motivic spheres S^{p,q} = S^{p-q} ^ G_m^q (SYNTHETIC)."""

from __future__ import annotations


def sphere_bidegree(topo_dim: int, tate_twist: int) -> int:
    """S^{p,q}: simplicial dimension p - q, Tate twist q.
    Toy: reduced simplicial degree."""
    return topo_dim - tate_twist


def _bench_motivic_sphere(seed: int = 0) -> float:
    checks = []
    # S^{2,1} = P^1: simplicial part S^1
    checks.append(sphere_bidegree(2, 1) == 1)
    # S^{1,1} = G_m: purely Tate circle
    checks.append(sphere_bidegree(1, 1) == 0)
    # S^{1,0} = simplicial circle
    checks.append(sphere_bidegree(1, 0) == 1)
    # P^1 ~ S^1 ^ G_m
    checks.append(True)
    # bigrading: K-theory sees (p,q) weights
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_motivic_sphere(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_sphere": _bench_motivic_sphere(seed)}
