"""Flow-insensitive escape analysis + scalar replacement on a tiny IR (SYNTHETIC).

Program = list of ops:
  ("alloc", name, [field_vals])      allocate object `name` w/ fields
  ("store", name, target)           object `name` escapes into `target`
  ("field", dst, name, i)           read field i of `name` into dst
  ("return", name)                  `name` escapes via return
An object is non-escaping (NoEscape) if it never reaches store/return or
is aliased into another escaping object. NoEscape allocs are replaced by
their fields (scalar replacement).
"""

from __future__ import annotations

_SEED = 20261231 + 897

Op = tuple[str, ...]


def escapes(prog: list[Op]) -> set[str]:
    """Set of allocation names that escape (fixpoint over aliasing)."""
    allocs: dict[str, list[str]] = {}
    alias: dict[str, str] = {}
    esc: set[str] = set()
    for op in prog:
        if op[0] == "alloc":
            allocs[op[1]] = list(op[2:])
        elif op[0] == "alias":
            alias[op[2]] = op[1]
        elif op[0] == "store":
            src, tgt = op[1], op[2]
            esc.add(alias.get(src, src))
            # fields of tgt that hold src also propagate the alias
            if tgt in allocs:
                for f in allocs[tgt]:
                    if f == src:
                        alias[f] = src
        elif op[0] == "return":
            esc.add(alias.get(op[1], op[1]))
    # propagate: any object reachable from an escaping field escapes
    changed = True
    while changed:
        changed = False
        for name, fields in allocs.items():
            if name in esc:
                for f in fields:
                    root = alias.get(f, f)
                    if root in allocs and root not in esc:
                        esc.add(root)
                        changed = True
    return esc


def scalar_replace(prog: list[Op]) -> list[Op]:
    """Drop non-escaping allocs; rewrite field reads to field values."""
    esc = escapes(prog)
    allocs = {op[1]: list(op[2:]) for op in prog if op[0] == "alloc"}
    out: list[Op] = []
    for op in prog:
        if op[0] == "alloc" and op[1] not in esc:
            continue  # scalar-replaced
        if op[0] == "field" and op[2] in allocs and op[2] not in esc:
            idx = int(op[3])
            out.append(("const", op[1], allocs[op[2]][idx]))
            continue
        out.append(op)
    return out


def _interp(prog: list[Op]) -> dict[str, str]:
    """Environment after running program (names -> values)."""
    env: dict[str, str] = {}
    allocs: dict[str, list[str]] = {}
    for op in prog:
        if op[0] == "alloc":
            allocs[op[1]] = list(op[2:])
            env[op[1]] = "<obj>"
        elif op[0] == "alias":
            env[op[2]] = env.get(op[1], op[1])
        elif op[0] == "field":
            env[op[1]] = allocs[op[2]][int(op[3])]
        elif op[0] == "const":
            env[op[1]] = op[2]
    return env


def bench_escape_analysis(seed: int = _SEED) -> dict[str, float]:
    prog: list[Op] = [
        ("alloc", "p", "10", "20"),  # p stays local
        ("field", "x", "p", "0"),
        ("field", "y", "p", "1"),
        ("alloc", "q", "30", "x"),
        ("store", "q", "global"),  # q escapes
        ("alloc", "r", "y", "q"),
        ("return", "r"),  # r escapes, and field q already escaping
    ]
    esc = escapes(prog)
    score = 0.0
    score += 1.0 if esc == {"q", "r"} else 0.0
    out = scalar_replace(prog)
    n_alloc = sum(1 for op in out if op[0] == "alloc")
    score += 1.0 if n_alloc == 2 else 0.0
    # semantics: env of rewritten program matches reference interp
    env_ref = _interp(prog)
    env_new = _interp(out)
    keys = set(env_ref) | set(env_new)
    score += (
        1.0
        if all(env_new.get(k) == env_ref.get(k) or env_ref.get(k) == "<obj>" for k in keys)
        else 0.0
    )
    # field reads resolved to constants
    score += 1.0 if ("const", "x", "10") in out and ("const", "y", "20") in out else 0.0
    return {"synthetic_escape_analysis": score / 4.0}
