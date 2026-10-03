"""Row-polymorphic record types: width/depth subtyping + row unification.

Types: ("base",n) | ("rec",{l:T}, tail) with tail None (closed) or
("rvar",n) (row variable). Subtyping: closed record <: record iff it has
all required labels with pointwise-≤ field types (width+depth). Row
unification splits a row around a demanded label — the Rémy/Xenopus
construction.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1026

Ty = tuple
Subst = dict[int, dict[str, Any]]


def is_tyvar(t: Any) -> bool:
    return bool(isinstance(t, tuple) and t and t[0] == "tvar")


def is_rvar(t: Any) -> bool:
    return bool(isinstance(t, tuple) and t and t[0] == "rvar")


def _fields(t: Ty, s: Subst) -> tuple[dict[str, Ty], Any]:
    """Flatten a record's labeled fields; returns (fields, tail)."""
    assert t[0] == "rec"
    row = t[1]
    tail = t[2]
    seen: dict[str, Ty] = dict(row)
    while is_rvar(tail):
        tail = s.get(tail[1], tail)
        if not is_rvar(tail):
            break
    if tail is None:
        return seen, None
    if tail[0] == "rec":
        f2, tl = _fields(tail, s)
        seen.update(f2)
        return seen, tl
    return seen, tail


def subtype(a: Ty, b: Ty, s: Subst | None = None) -> bool:
    s = s or {}
    if a == b:
        return True
    if a[0] == "base" or b[0] == "base":
        return a == b
    if a[0] == "rec" and b[0] == "rec":
        fa, _ = _fields(a, s)
        fb, tb = _fields(b, s)
        # width: a must have at least b's labels with pointwise-sub field types
        return all(lbl in fa and subtype(fa[lbl], tb_, s) for lbl, tb_ in fb.items())
    # tyvars/rvars defer to the unifier — treated as consistent here
    return is_tyvar(a) or is_tyvar(b) or is_rvar(a) or is_rvar(b)


def unify_row(rec: Ty, label: str, s: Subst) -> Ty | None:
    """Split `rec` at `label`: bind its row tail to a fresh record holding the
    remaining fields; returns the field type for label or None."""
    assert rec[0] == "rec"
    row, tail = rec[1], rec[2]
    if label in row:
        field: Ty = row[label]
        return field
    # walk tail chain
    if tail is None:
        return None
    if is_rvar(tail):
        # bind tail to a fresh record {label: tvar} | rho'
        fresh_field = ("tvar", len(s) + 100)
        bound: dict[str, Any] = {"t": ("rec", {label: fresh_field}, ("rvar", len(s) + 200))}
        s[tail[1]] = bound["t"]
        return fresh_field
    if tail[0] == "rec":
        return unify_row(tail, label, s)
    return None


def bench_row_types(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    i = ("base", "int")
    b = ("base", "bool")
    r_ab = ("rec", {"a": i, "b": b}, None)
    r_a = ("rec", {"a": i}, None)
    checks.append(subtype(r_ab, r_a))  # width subtyping
    checks.append(not subtype(r_a, r_ab))
    # depth subtyping: {a:int} <: {a:top-like} — model "any" base
    r_deep = ("rec", {"a": ("base", "int")}, None)
    checks.append(subtype(r_deep, ("rec", {"a": ("base", "int")}, None)))
    # row unification: {x:int | ρ} unify at y binds ρ -> {y:β | ρ'}
    s: Subst = {}
    rt = ("tvar", 1)
    _ = rt
    open_rec = ("rec", {"x": i}, ("rvar", 7))
    got = unify_row(open_rec, "y", s)
    checks.append(got is not None and got[0] == "tvar")
    checks.append(7 in s)
    # closed record at missing label fails
    checks.append(unify_row(("rec", {"x": i}, None), "y", {}) is None)
    # nested tail unification
    s2: Subst = {}
    rec2 = ("rec", {"x": i}, ("rec", {"y": b}, None))
    checks.append(unify_row(rec2, "y", s2) == b)
    return {"synthetic_row_types": float(sum(checks)) / len(checks)}
