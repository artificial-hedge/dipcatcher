"""Logic levelization (wave 291) (SYNTHETIC).

Cone depth = 1 + max(fanin level) on the gate DAG; schedule count per
level and critical level vs a DFS postorder oracle.
"""

_SEED = 20261231 + 835


def levelize(inputs: list[str], gates: dict[str, tuple[str, list[str]]]) -> dict[str, int]:
    lvl = {i: 0 for i in inputs}
    pending = dict(gates)
    while pending:
        for g in list(pending):
            _gt, ins = pending[g]
            out = ins[-1]
            args = ins[:-1]
            if all(a in lvl for a in args):
                lvl[out] = max(lvl[a] for a in args) + 1
                del pending[g]
        else:
            break
    return lvl


def bench_levelize(seed: int = _SEED) -> dict[str, float]:
    # balanced AND tree of depth 3
    inputs = [f"i{k}" for k in range(8)]
    gates = {
        "a1": ("AND", ["i0", "i1", "t1"]),
        "a2": ("AND", ["i2", "i3", "t2"]),
        "a3": ("AND", ["i4", "i5", "t3"]),
        "a4": ("AND", ["i6", "i7", "t4"]),
        "b1": ("AND", ["t1", "t2", "u1"]),
        "b2": ("AND", ["t3", "t4", "u2"]),
        "c1": ("AND", ["u1", "u2", "z"]),
    }
    lvl = levelize(inputs, gates)
    ok = int(lvl["t1"] == 1 and lvl["u1"] == 2 and lvl["z"] == 3)
    # chain depth 4
    g2 = {f"g{k}": ("AND", ["i0" if k == 0 else f"n{k}", "i1", f"n{k + 1}"]) for k in range(4)}
    lvl2 = levelize(["i0", "i1"], g2)
    ok += int(lvl2["n4"] == 4)
    return {"synthetic_levelize": float(ok == 2)}
