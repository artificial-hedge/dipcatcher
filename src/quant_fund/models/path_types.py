"""Path types and transport in a finite type model (SYNTHETIC).

Types are finite sets of elements; a path p : x =_A y is a walk through
A's identity data (modeled as an equality witness with a proof tag).
transport moves an element of P(x) to P(y) along p; J is the path
induction eliminator (reduces to the motive applied at refl).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SEED = 20261231 + 1053

Path = tuple[str, Any, Any]  # ("refl", x, x) | ("trans", ...) composition tag


def refl(x: Any) -> Path:
    return ("refl", x, x)


def sym(p: Path) -> Path:
    return ("sym", p[2], p[1])


def trans(p: Path, q: Path) -> Path:
    """p : x=y, q : y=z -> x=z."""
    if p[2] != q[1]:
        raise ValueError("endpoints don't match")
    if p[0] == "refl":
        return q
    if q[0] == "refl":
        return p
    return ("trans", p, q)


def path_endpoints(p: Path) -> tuple[Any, Any]:
    if p[0] == "refl":
        return p[1], p[2]
    if p[0] == "sym":
        return p[2], p[1]
    return (p[1][1] if p[0] == "trans" else p[1], p[2])


def is_path(p: Path) -> bool:
    """Well-formedness: composition paths connect correctly."""
    if p[0] == "refl":
        return bool(p[1] == p[2])
    if p[0] == "sym":
        return True
    if p[0] == "trans":
        q, r = p[1], p[2]
        return bool(q[2] == r[1])
    return False


def transport(p: Path, family: Callable[[Any], Any], elem: Any) -> Any:
    """transport^P(p) : P(x) -> P(y). On finite model P maps elements;
    transport evaluates the family at the endpoint — equal types share
    elements so transport is identity on the carrier."""
    if p[0] == "refl":
        return elem
    # family(x) and family(y) are the same type along p; element moves identically
    return family(p[2]) if callable(family) else elem


def j_elim(motive: Callable[..., Any], base_case: Any, p: Path) -> Any:
    """J eliminator: to prove C(x,y,p) for all paths it suffices to prove
    C(x,x,refl). Returns base_case when p reduces to refl; else applies
    the motive using the path's evidence."""
    if p[0] == "refl":
        return base_case
    return motive(p)


def bench_path_types(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    r = refl(3)
    checks.append(is_path(r) and path_endpoints(r) == (3, 3))
    # trans of refl's collapses
    p = trans(refl(1), refl(1))
    checks.append(p == refl(1) or is_path(p))
    # sym reverses endpoints
    s = sym(refl(2))
    checks.append(path_endpoints(s) == (2, 2))
    # transport along refl is identity
    checks.append(transport(refl(5), lambda x: x, "elem") == "elem")
    # J reduces to base case on refl
    checks.append(j_elim(lambda p: "mot", "base", refl(0)) == "base")
    # non-refl path goes through motive
    checks.append(j_elim(lambda p: ("mot", p[0]), "base", sym(refl(1))) == ("mot", "sym"))
    # malformed trans raises
    try:
        trans(("refl", 1, 1), ("refl", 2, 2))
        checks.append(False)
    except ValueError:
        checks.append(True)
    return {"synthetic_path_types": float(sum(checks)) / len(checks)}
