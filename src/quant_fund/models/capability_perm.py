"""Permission/capability types for alias control (SYNTHETIC).

Permissions: U (unique, split/join), S (shared, immutable through alias),
R (readonly view). State maps place -> perm. Operations:
("split",x,x1,x2): unique x splits into two exclusive aliases that are
unusable until joined; ("join",x1,x2,x): recombines; ("alias_ro",x,y):
readonly view coexisting with owner; ("write",x): requires U or owning;
("read",x): any permission allows.
"""

from __future__ import annotations

_SEED = 20261231 + 1039

Op = tuple


class PermError(Exception):
    pass


def run(ops: list[Op]) -> dict[str, str]:
    st: dict[str, str] = {}
    for op in ops:
        tag = op[0]
        if tag == "alloc":
            st[op[1]] = "U"
        elif tag == "split":
            _, x, a, b = op
            if st.get(x) != "U":
                raise PermError(f"split needs unique {x}")
            del st[x]
            st[a] = "U_split"
            st[b] = "U_split"
        elif tag == "join":
            _, a, b, x = op
            if st.get(a) != "U_split" or st.get(b) != "U_split":
                raise PermError("join needs both split halves")
            del st[a]
            del st[b]
            st[x] = "U"
        elif tag == "alias_ro":
            _, x, y = op
            if st.get(x) not in ("U", "S", "R"):
                raise PermError(f"alias_ro needs live owner {x}")
            st[y] = "R"
        elif tag == "drop_ro":
            del st[op[1]]
        elif tag == "write":
            x = op[1]
            if st.get(x) != "U":
                raise PermError(f"write needs unique, got {st.get(x)}")
            if any(p == "R" for k, p in st.items() if k != x):
                pass  # readonly views on OTHER places are fine
        elif tag == "read":
            if st.get(op[1]) is None:
                raise PermError(f"read dead {op[1]}")
        else:
            raise PermError(op)
    return st


def bench_capability_perm(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # alloc -> split -> write fails (split halves aren't writable) -> join -> write ok
    st = run([("alloc", "x"), ("split", "x", "a", "b"), ("join", "a", "b", "x"), ("write", "x")])
    checks.append(st.get("x") == "U")
    try:
        run([("alloc", "x"), ("split", "x", "a", "b"), ("write", "a")])
        ok1 = False
    except PermError:
        ok1 = True
    checks.append(ok1)
    # alias_ro then read ok, write blocked
    st2 = run([("alloc", "x"), ("alias_ro", "x", "y"), ("read", "y")])
    checks.append(st2.get("y") == "R")
    try:
        run([("alloc", "x"), ("alias_ro", "x", "y"), ("write", "y")])
        ok2 = False
    except PermError:
        ok2 = True
    checks.append(ok2)
    # owner stays writable while readonly alias exists (non-lexical: alias dies at last read)
    st3 = run([("alloc", "x"), ("alias_ro", "x", "y"), ("read", "y"), ("write", "x")])
    checks.append(st3.get("x") == "U")
    # split without join then write original var: dead -> error
    try:
        run([("alloc", "x"), ("split", "x", "a", "b"), ("write", "x")])
        ok3 = False
    except PermError:
        ok3 = True
    checks.append(ok3)
    return {"synthetic_capability_perm": float(sum(checks)) / len(checks)}
