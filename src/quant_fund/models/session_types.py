"""Binary session types: duality + a checker for a mini process language.

Types: ("end",) | ("send",T,S) | ("recv",T,S) | ("sel",{l:S}) internal
choice | ("bra",{l:S}) external choice | ("mu",X,S) | ("var",X).
Processes: ("send",ch,v,P) | ("recv",ch,x,P) | ("sel",ch,l,P) |
("offer",ch,{l:P}) | ("end",). The checker simulates the type-level
protocol — a protocol is type-correct iff every action follows the
prescribed type, and dual types interlock by construction.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1027


def dual(s: Any) -> Any:
    tag = s[0]
    if tag == "send":
        return ("recv", s[1], dual(s[2]))
    if tag == "recv":
        return ("send", s[1], dual(s[2]))
    if tag == "sel":
        return ("bra", {lbl: dual(t) for lbl, t in s[1].items()})
    if tag == "bra":
        return ("sel", {lbl: dual(t) for lbl, t in s[1].items()})
    if tag == "mu":
        return ("mu", s[1], dual(s[2]))
    return s  # end/var


def unfold(s: Any) -> Any:
    while s[0] == "mu":
        s = _subst(s[2], s[1], s)
    return s


def _subst(s: Any, x: str, r: Any) -> Any:
    if s[0] == "var" and s[1] == x:
        return r
    if s[0] in ("send", "recv"):
        return (s[0], s[1], _subst(s[2], x, r))
    if s[0] in ("sel", "bra"):
        return (s[0], {lbl: _subst(t, x, r) for lbl, t in s[1].items()})
    return s


class SessionError(Exception):
    pass


def check(p: Any, s: Any) -> None:
    while True:
        s = unfold(s)
        pt = p[0]
        st = s[0]
        if pt == "end":
            if st != "end":
                raise SessionError(f"open channel at end: {s}")
            return
        if pt == "send":
            if st != "send":
                raise SessionError(f"send vs {st}")
            p, s = p[3], s[2]
            continue
        if pt == "recv":
            if st != "recv":
                raise SessionError(f"recv vs {st}")
            p, s = p[3], s[2]
            continue
        if pt == "sel":
            if st != "sel":
                raise SessionError(f"select vs {st}")
            lbl = p[2]
            if lbl not in s[1]:
                raise SessionError(f"label {lbl}")
            p, s = p[3], s[1][lbl]
            continue
        if pt == "offer":
            if st != "bra":
                raise SessionError(f"offer vs {st}")
            for lbl, cont in p[2].items():
                if lbl not in s[1]:
                    raise SessionError(f"offer label {lbl}")
                check(cont, s[1][lbl])
            return
        raise SessionError(f"bad proc {pt}")


def bench_session_types(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    n = ("base", "int")
    # client: !int. !int. ?int. end ; server: dual
    cli_ty = ("send", n, ("send", n, ("recv", n, ("end",))))
    srv_ty = dual(cli_ty)
    checks.append(srv_ty == ("recv", n, ("recv", n, ("send", n, ("end",)))))
    client = ("send", "c", 1, ("send", "c", 2, ("recv", "c", "s", ("end",))))
    server = ("recv", "c", "a", ("recv", "c", "b", ("send", "c", 0, ("end",))))
    try:
        check(client, cli_ty)
        check(server, srv_ty)
        ok = True
    except SessionError:
        ok = False
    checks.append(ok)
    # protocol violation: client sends when type says recv
    bad = ("recv", "c", "a", ("end",))
    try:
        check(bad, cli_ty)
        ok2 = False
    except SessionError:
        ok2 = True
    checks.append(ok2)
    # recursive type: mu X. !int.X with sel/bra — client loops send forever;
    # single unfold step exercised via a finite prefix
    loop_ty = ("mu", "X", ("send", n, ("var", "X")))
    unf = unfold(loop_ty)
    checks.append(unf[0] == "send" and unf[2][0] == "mu")
    # choice: sel{add:...} matches
    sel_ty = ("sel", {"add": ("send", n, ("end",))})
    proc = ("sel", "c", "add", ("send", "c", 1, ("end",)))
    try:
        check(proc, sel_ty)
        ok3 = True
    except SessionError:
        ok3 = False
    checks.append(ok3)
    return {"synthetic_session_types": float(sum(checks)) / len(checks)}
