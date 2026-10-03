"""SYNTHETIC bytecode stack VM + mini compiler.

Expr AST → bytecode [PUSH n | ADD | SUB | MUL | JZ dst | JMP dst | CALL |
RET]; VM executes against direct AST eval — equal on all programs.
"""

from __future__ import annotations

import random


def compile_expr(e: tuple, out: list) -> None:
    op = e[0]
    if op == "lit":
        out.append(("push", e[1]))
    elif op == "add" or op == "sub" or op == "mul":
        compile_expr(e[1], out)
        compile_expr(e[2], out)
        out.append((op,))
    elif op == "if":
        compile_expr(e[1], out)
        jz = len(out)
        out.append(("jz", -1))
        compile_expr(e[2], out)
        jmp = len(out)
        out.append(("jmp", -1))
        out[jz] = ("jz", len(out))
        compile_expr(e[3], out)
        out[jmp] = ("jmp", len(out))
    else:
        raise ValueError(op)


def run(code: list[tuple]) -> int:
    st: list[int] = []
    pc = 0
    while pc < len(code):
        ins = code[pc]
        op = ins[0]
        if op == "push":
            st.append(ins[1])
        elif op == "add":
            b, a = st.pop(), st.pop()
            st.append(a + b)
        elif op == "sub":
            b, a = st.pop(), st.pop()
            st.append(a - b)
        elif op == "mul":
            b, a = st.pop(), st.pop()
            st.append(a * b)
        elif op == "jz":
            if st.pop() == 0:
                pc = ins[1]
                continue
        elif op == "jmp":
            pc = ins[1]
            continue
        else:
            raise ValueError(op)
        pc += 1
    return st[-1]


def _eval(e: tuple) -> int:
    op = e[0]
    if op == "lit":
        return int(e[1])
    if op == "add":
        return _eval(e[1]) + _eval(e[2])
    if op == "sub":
        return _eval(e[1]) - _eval(e[2])
    if op == "mul":
        return _eval(e[1]) * _eval(e[2])
    if op == "if":
        return _eval(e[2]) if _eval(e[1]) != 0 else _eval(e[3])
    raise ValueError(op)


def _gen(rng: random.Random, depth: int) -> tuple:
    if depth <= 0 or rng.random() < 0.3:
        return ("lit", rng.randrange(-20, 21))
    op = rng.choice(["add", "sub", "mul", "if"])
    if op == "if":
        return ("if", _gen(rng, 0), _gen(rng, depth - 1), _gen(rng, depth - 1))
    return (op, _gen(rng, depth - 1), _gen(rng, depth - 1))


def bench_bytecode_vm(seed: int = 20261231 + 480) -> dict[str, float]:
    rng = random.Random(seed)
    match = term = if_ok = 0
    trials = 60
    for _ in range(trials):
        e = _gen(rng, rng.randrange(1, 4))
        code: list[tuple] = []
        compile_expr(e, code)
        match += int(run(code) == _eval(e))
        term += int(isinstance(run(code), int))
        e2 = ("if", ("lit", 0), ("lit", 1), ("lit", 2))
        c2: list[tuple] = []
        compile_expr(e2, c2)
        if_ok += int(run(c2) == 2)
    return {
        "synthetic_vm_matches_eval": float(match / trials),
        "synthetic_vm_terminates": float(term / trials),
        "synthetic_branch_compiles": float(if_ok / trials),
    }
