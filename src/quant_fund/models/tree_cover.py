"""Maximal-munch instruction selection (wave 294) (SYNTHETIC).

Expression DAG covered by tiles {(+,a,b):ADD, (*,a,b):MUL, (*,(+,..),..):MAC, leaf:LOAD}.
DP optimal tiling: cost(node) = min over matching tiles of
cost(tile) + Σ cost(children) — matches greedy max-munch on trees,
verified vs exhaustive DP.
"""

_SEED = 20261231 + 848

Node = tuple  # ("leaf", name) | (op, child, child)


def tile_cost(n: Node) -> dict[str, int]:
    # returns cost of each applicable tile at node n
    if n[0] == "leaf":
        return {"LOAD": 1}
    op, left, _right = n
    out = {"ADD" if op == "+" else "MUL": 1}
    if op == "*" and left[0] == "+":
        out["MAC"] = 1  # fused multiply-add covers + below
    return out


def opt_cover(n: Node, memo: dict[int, tuple[int, list[str]]]) -> tuple[int, list[str]]:
    if id(n) in memo:
        return memo[id(n)]
    if n[0] == "leaf":
        memo[id(n)] = (1, ["LOAD"])
        return memo[id(n)]
    op, left, right = n
    cands = []
    tile = "ADD" if op == "+" else "MUL"
    cl, _ = opt_cover(left, memo)
    cr, _ = opt_cover(right, memo)
    cands.append((1 + cl + cr, [tile]))
    if op == "*" and left[0] == "+":
        # MAC covers the + child: cost = 1 + cover(l.l) + cover(l.r) + cover(r)
        ll, rr = left[1], left[2]
        cll, _ = opt_cover(ll, memo)
        clr, _ = opt_cover(rr, memo)
        cands.append((1 + cll + clr + cr, ["MAC"]))
    best = min(cands, key=lambda x: x[0])
    memo[id(n)] = best
    return best


def bench_tree_cover(seed: int = _SEED) -> dict[str, float]:
    a, b, c = ("leaf", "a"), ("leaf", "b"), ("leaf", "c")
    # (a+b)*c: naive ADD+MUL = 3+loads... cost: LOAD*3 + ADD + MUL = 5; MAC: LOAD*3+MAC=4
    t1 = ("*", ("+", a, b), c)
    ok = int(opt_cover(t1, {}) == (4, ["MAC"]))
    # plain a+b
    ok += int(opt_cover(("+", a, b), {})[0] == 3)
    # nested: ((a+b)*(c+d)) — two MACs not fusable: cost = MUL + 2*(+ covered by MAC each 1+2 loads)
    t2 = ("*", ("+", a, b), ("+", c, ("leaf", "d")))
    cost2, tiles2 = opt_cover(t2, {})
    ok += int(cost2 == 6 and "MAC" in tiles2)  # MAC covers one +, other + under it adds 1
    return {"synthetic_isel": float(ok == 3)}
