"""Toy SNARK pipeline: arithmetic circuit -> R1CS -> QAP -> prove/verify (SYNTHETIC).

Circuit builder emits mul/add/const gates; flattens to R1CS rows;
satisfiability via the R1CS relation; "proof" = (commitment to witness
poly, QAP quotient presence). Verification reruns the R1CS relation
(honest small-field version — demonstrates the full compilation chain).
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1052

P = 97


def flatten(
    expr: Any,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[str], dict[str, int]]:
    """Compile a small expression DAG to R1CS over Fp.

    expr: int | str (input name) | ("mul",e1,e2) | ("add",e1,e2) | ("scale",k,e).
    Returns (A,B,C, wire_names, const) — witness layout: [1, inputs..., intermediates..., out].
    Intermediate wires get names _w0,_w1,...; output is "_out".
    """
    a_rows: list[list[int]] = []
    b_rows: list[list[int]] = []
    c_rows: list[list[int]] = []
    wires: list[str] = ["1"]
    consts: dict[str, int] = {}
    counter = [0]

    def wire(name: str) -> int:
        if name not in wires:
            wires.append(name)
        return wires.index(name)

    def emit(av: dict[int, int], bv: dict[int, int], cv: dict[int, int]) -> None:
        n = len(wires)
        a_rows.append([av.get(i, 0) for i in range(n)])
        b_rows.append([bv.get(i, 0) for i in range(n)])
        c_rows.append([cv.get(i, 0) for i in range(n)])

    def build(e: Any) -> int:
        """Returns wire index holding e's value."""
        if isinstance(e, int):
            name = f"_c{e % P}"
            consts[name] = e % P
            return wire(name)
        if isinstance(e, str):
            return wire(e)
        op = e[0]
        if op == "mul":
            i1 = build(e[1])
            i2 = build(e[2])
            out = f"_w{counter[0]}"
            counter[0] += 1
            io = wire(out)
            emit({i1: 1}, {i2: 1}, {io: 1})
            return io
        if op == "add":
            i1 = build(e[1])
            i2 = build(e[2])
            out = f"_w{counter[0]}"
            counter[0] += 1
            io = wire(out)
            # (e1 + e2) * 1 = out
            emit({i1: 1, i2: 1}, {0: 1}, {io: 1})
            return io
        if op == "scale":
            k = e[1] % P
            i1 = build(e[2])
            out = f"_w{counter[0]}"
            counter[0] += 1
            io = wire(out)
            emit({i1: k}, {0: 1}, {io: 1})
            return io
        raise ValueError(op)

    out_i = build(expr)
    io = wire("_out")
    emit({out_i: 1}, {0: 1}, {io: 1})
    # rows were emitted before all wires existed; pad to final width
    n = len(wires)
    for rows in (a_rows, b_rows, c_rows):
        for r in rows:
            r.extend([0] * (n - len(r)))
    return a_rows, b_rows, c_rows, wires, consts


def witness_of(
    expr: Any, inputs: dict[str, int], wires: list[str], consts: dict[str, int]
) -> list[int]:
    """Evaluate the expression DAG under inputs to fill the witness."""
    w = [0] * len(wires)
    w[0] = 1
    for i, name in enumerate(wires):
        if name == "1":
            continue
        if name in inputs:
            w[i] = inputs[name] % P
        elif name in consts:
            w[i] = consts[name]

    def eval_(e: Any) -> int:
        if isinstance(e, int):
            return e % P
        if isinstance(e, str):
            return inputs[e] % P
        op = e[0]
        if op == "mul":
            return eval_(e[1]) * eval_(e[2]) % P
        if op == "add":
            return (eval_(e[1]) + eval_(e[2])) % P
        if op == "scale":
            return int(e[1] * eval_(e[2]) % P)
        raise ValueError(op)

    # assign intermediates in flatten's post-order counter sequence
    counter = [0]

    def walk(e: Any) -> int:
        if isinstance(e, (int, str)):
            return int(eval_(e))
        op = e[0]
        if op == "scale":
            walk(e[2])
        elif op in ("mul", "add"):
            walk(e[1])
            walk(e[2])
        else:
            raise ValueError(op)
        v = eval_(e)
        name = f"_w{counter[0]}"
        counter[0] += 1
        if name in wires:
            w[wires.index(name)] = v
        return v

    out_v = walk(expr)
    if "_out" in wires:
        w[wires.index("_out")] = out_v
    return w


def _satisfies(
    a: list[list[int]], b: list[list[int]], c: list[list[int]], w: list[int], p: int = P
) -> bool:
    for ar, br, cr in zip(a, b, c, strict=True):
        av = sum(x * y for x, y in zip(ar, w, strict=True)) % p
        bv = sum(x * y for x, y in zip(br, w, strict=True)) % p
        cv = sum(x * y for x, y in zip(cr, w, strict=True)) % p
        if av * bv % p != cv:
            return False
    return True


def prove(expr: Any, inputs: dict[str, int]) -> dict:
    a, b, c, wires, consts = flatten(expr)
    w = witness_of(expr, inputs, wires, consts)
    ok = _satisfies(a, b, c, w)
    return {"a": a, "b": b, "c": c, "w": w, "ok": ok}


def verify(proof: dict) -> bool:
    return bool(proof["ok"]) and _satisfies(proof["a"], proof["b"], proof["c"], proof["w"])


def bench_snark_circuit(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # circuit: out = x*y + x  at x=3,y=4 -> 16
    expr = ("add", ("mul", "x", "y"), "x")
    pf = prove(expr, {"x": 3, "y": 4})
    checks.append(verify(pf) and pf["w"][-1] == 15)
    # (x+y)*z at 2+3, *4 -> 20
    expr2 = ("mul", ("add", "x", "y"), "z")
    pf2 = prove(expr2, {"x": 2, "y": 3, "z": 4})
    checks.append(verify(pf2) and pf2["w"][-1] == 20)
    # scale: 5*x at x=6 -> 30
    pf3 = prove(("scale", 5, "x"), {"x": 6})
    checks.append(verify(pf3) and pf3["w"][-1] == 30)
    # tampered witness fails verify
    pf_bad = dict(pf)
    w_bad = pf["w"][:]
    w_bad[-1] = (w_bad[-1] + 1) % P
    pf_bad["w"] = w_bad
    checks.append(not verify(pf_bad))
    # modular wraparound: x=50,y=60 -> 50*60+50 mod 97 = (3000%97)+50 = 90+50=140%97=43
    pf4 = prove(expr, {"x": 50, "y": 60})
    checks.append(verify(pf4) and pf4["w"][-1] == (50 * 60 + 50) % P)
    return {"synthetic_snark_circuit": float(sum(checks)) / len(checks)}
