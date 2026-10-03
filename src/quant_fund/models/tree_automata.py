"""Bottom-up tree automaton over ranked trees; acceptance + counting."""

import numpy as np

_SEED = 20261231 + 550

Tree = str | tuple[str, "Tree", "Tree"] | tuple[str, "Tree"]


def eval_tree(t: Tree, rules: dict[tuple[str, tuple[str, ...]], str], leaf: dict[str, str]) -> str:
    """Bottom-up state assignment; returns state at root."""
    if isinstance(t, str):
        return leaf[t]
    name = t[0]
    child_states = tuple(eval_tree(c, rules, leaf) for c in t[1:])
    return rules[(name, child_states)]


def accepts(
    t: Tree, rules: dict[tuple[str, tuple[str, ...]], str], leaf: dict[str, str], finals: set[str]
) -> bool:
    return eval_tree(t, rules, leaf) in finals


def _gen_tree(rng: np.random.RandomState, depth: int) -> Tree:
    if depth == 0 or rng.random() < 0.3:
        return str(rng.choice(["0", "1"]))
    op = rng.choice(["and", "or", "not"])
    if op == "not":
        return ("not", _gen_tree(rng, depth - 1))
    return (op, _gen_tree(rng, depth - 1), _gen_tree(rng, depth - 1))


def _truth(t: Tree) -> int:
    if isinstance(t, str):
        return int(t)
    name, *children = t
    if name == "and":
        return _truth(children[0]) & _truth(children[1])
    if name == "or":
        return _truth(children[0]) | _truth(children[1])
    return 1 - _truth(children[0])


def _rules() -> tuple[dict, dict, set[str]]:
    leaf = {"0": "q0", "1": "q1"}
    rules: dict = {}
    for a in ("q0", "q1"):
        for b in ("q0", "q1"):
            rules[("and", (a, b))] = "q1" if (a == b == "q1") else "q0"
            rules[("or", (a, b))] = "q1" if (a == "q1" or b == "q1") else "q0"
    rules[("not", ("q0",))] = "q1"
    rules[("not", ("q1",))] = "q0"
    return rules, leaf, {"q1"}


def bench_tree_automata(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    rules, leaf, finals = _rules()
    n = 200
    acc = 0
    for _ in range(n):
        t = _gen_tree(rng, 3)
        acc += int(accepts(t, rules, leaf, finals) == bool(_truth(t)))
    return {"synthetic_acceptance_exact": float(acc / n)}
