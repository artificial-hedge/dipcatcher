"""Peephole optimizer: local rewrite rules over a stack-machine program."""

import numpy as np

_SEED = 20261231 + 723

# instrs: ("push", v) | ("load", name) | "add" | "mul" | "neg"
Instr = tuple | str


def peephole(prog: list[Instr]) -> list[Instr]:
    out: list[Instr] = []
    for ins in prog:
        out.append(ins)
        # fold const-op-const
        if len(out) >= 3 and out[-1] in ("add", "mul"):
            a, b, op = out[-3], out[-2], out[-1]
            if isinstance(a, tuple) and a[0] == "push" and isinstance(b, tuple) and b[0] == "push":
                v = a[1] + b[1] if op == "add" else a[1] * b[1]
                out = out[:-3] + [("push", v)]
                continue
        # x*0 -> push 0 ; x*1 -> x ; x+0 -> x
        if len(out) >= 2 and out[-1] == "mul" and out[-2] == ("push", 0):
            out = out[:-2] + ["pop", ("push", 0)]
        elif (
            len(out) >= 2
            and out[-1] == "mul"
            and out[-2] == ("push", 1)
            or len(out) >= 2
            and out[-1] == "add"
            and out[-2] == ("push", 0)
            or len(out) >= 2
            and out[-1] == "neg"
            and out[-2] == "neg"
        ):
            out = out[:-2]
    return out


def run_prog(prog: list[Instr], env: dict[str, int]) -> int:
    st: list[int] = []
    for ins in prog:
        if ins == "add":
            b, a = st.pop(), st.pop()
            st.append(a + b)
        elif ins == "mul":
            b, a = st.pop(), st.pop()
            st.append(a * b)
        elif ins == "neg":
            st.append(-st.pop())
        elif ins == "pop":
            st.pop()
        else:
            st.append(env[ins[1]] if ins[0] == "load" else int(ins[1]))
    return st[-1]


def bench_peephole_opt(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        prog: list[Instr] = []
        depth = 0
        for _ in range(rng.randint(3, 9)):
            r = rng.rand()
            if r < 0.4 or depth == 0:
                prog.append(("push", int(rng.randint(-4, 4))))
                depth += 1
            elif r < 0.6:
                prog.append(("load", "x"))
                depth += 1
            elif depth >= 2:
                prog.append(rng.choice(["add", "mul"]))
                depth -= 1
            elif depth >= 1:
                prog.append("neg")
        # ensure a stack value survives
        prog += [("push", int(rng.randint(1, 5))), "add"]
        opt = peephole(prog)
        ok += float(run_prog(opt, {"x": 7}) == run_prog(prog, {"x": 7}))
    return {"synthetic_peephole_preserves": ok / trials}
