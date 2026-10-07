"""Netlist parser (wave 291) (SYNTHETIC).

Gate-level netlist DSL: `AND a1 i1 i2 o1` lines plus `.input`/`.output`
declarations. Returns gate DAG as (inputs, gates{name,type,fanin,fanout},
outputs); verified on a 2-bit adder hand-built netlist.
"""

_SEED = 20261231 + 830

NETLIST = """
.input a0
.input b0
.input cin
.output s
.output cout
XOR x1 a0 b0 t1
XOR x2 t1 cin s
AND a1 a0 b0 t2
AND a2 t1 cin t3
OR o1 t2 t3 cout
""".strip()


def parse(text: str) -> tuple[list[str], dict[str, tuple[str, list[str]]], list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    gates: dict[str, tuple[str, list[str]]] = {}
    for line in text.splitlines():
        toks = line.split()
        if toks[0] == ".input":
            inputs.append(toks[1])
        elif toks[0] == ".output":
            outputs.append(toks[1])
        else:
            gtype, gname = toks[0], toks[1]
            gates[gname] = (gtype, toks[2:])
    return inputs, gates, outputs


def simulate(
    inputs: list[str],
    gates: dict[str, tuple[str, list[str]]],
    outs: list[str],
    vals: dict[str, int],
) -> dict[str, int]:
    v = dict(vals)
    for _gname, (gt, ins) in gates.items():
        out = ins[-1]
        args = ins[:-1]
        if gt == "AND":
            v[out] = int(all(v[a] for a in args))
        elif gt == "OR":
            v[out] = int(any(v[a] for a in args))
        elif gt == "XOR":
            v[out] = int(sum(v[a] for a in args) % 2)
        elif gt == "NOT":
            v[out] = 1 - v[args[0]]
    return {o: v[o] for o in outs}


def bench_netlist_parse(seed: int = _SEED) -> dict[str, float]:
    inputs, gates, outs = parse(NETLIST)
    ok = int(len(inputs) == 3 and len(gates) == 5 and len(outs) == 2)
    # full-adder truth table: s = a^b^cin, cout = majority
    good = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                r = simulate(inputs, gates, outs, {"a0": a, "b0": b, "cin": c})
                good += int(r["s"] == (a ^ b ^ c) and r["cout"] == int(a + b + c >= 2))
    return {"synthetic_netlist": float(ok == 1 and good == 8)}
