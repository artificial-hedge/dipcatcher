"""Finite-model-property + compactness-lite checks on FO sentences (SYNTHETIC)."""

from __future__ import annotations


def satisfies(struct: dict, sent: tuple) -> bool:
    """sentences over domain: ("all",var,f), ("ex",var,f), ("edge",a,b), ("and",f,g),
    ("not",f), ("imp",f,g), ("refl",), ("trans",), ("symm",)."""
    tag = sent[0]
    n = struct["n"]
    if tag == "refl":
        return all((x, x) in struct["edge"] for x in range(n))
    if tag == "symm":
        return all((y, x) in struct["edge"] for (x, y) in struct["edge"])
    if tag == "trans":
        return all(
            (x, z) in struct["edge"]
            for (x, y) in struct["edge"]
            for (x2, z) in struct["edge"]
            if y == x2
        )
    if tag == "edge":
        return (sent[1], sent[2]) in struct["edge"]
    if tag == "and":
        return satisfies(struct, sent[1]) and satisfies(struct, sent[2])
    if tag == "imp":
        return (not satisfies(struct, sent[1])) or satisfies(struct, sent[2])
    if tag == "not":
        return not satisfies(struct, sent[1])
    raise ValueError(tag)


def finite_model(sent: tuple, max_n: int = 4) -> int | None:
    """Smallest structure size (toy enumeration over edge-sets) satisfying sent; None if none."""
    import itertools

    for n in range(1, max_n + 1):
        edges = [(x, y) for x in range(n) for y in range(n)]
        # only try symmetric closures for graph-like sentences; small n -> all 2^|E| too big, sample k edges
        for k in range(0, min(len(edges), 4) + 1):
            for sub in itertools.combinations(edges, k):
                if satisfies({"n": n, "edge": set(sub)}, sent):
                    return n
    return None


def _bench_compactness_lite(seed: int = 0) -> float:
    checks = []
    st = {"n": 3, "edge": {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}}
    checks.append(satisfies(st, ("refl",)))
    checks.append(satisfies(st, ("symm",)))
    st_nt = {"n": 3, "edge": {(0, 1), (1, 2)}}
    checks.append(not satisfies(st_nt, ("trans",)))
    # transitivity holds on full relation
    full = {"n": 2, "edge": {(0, 0), (0, 1), (1, 0), (1, 1)}}
    checks.append(satisfies(full, ("trans",)))
    # smallest model of "symm" is n=1 with empty edge set (vacuous)
    checks.append(finite_model(("symm",)) == 1)
    # smallest model of refl is n=1 with self loop
    checks.append(finite_model(("refl",)) is not None)
    return sum(checks) / len(checks)


def bench_compactness_lite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compactness_lite": _bench_compactness_lite(seed)}
