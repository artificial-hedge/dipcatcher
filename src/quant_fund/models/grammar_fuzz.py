"""Grammar-based fuzzer: generate valid expressions from a CFG."""

import numpy as np

_SEED = 20261231 + 675

_RULES = {
    "E": [["E", "+", "T"], ["T"]],
    "T": [["T", "*", "F"], ["F"]],
    "F": [["(", "E", ")"], ["n"]],
}


def gen(sym: str, rng: np.random.RandomState, depth: int = 0) -> list[str]:
    if depth > 4:
        choices = [r for r in _RULES[sym] if "n" in r or len(r) == 1]
        prod = choices[rng.randint(len(choices))]
    else:
        prod = _RULES[sym][rng.randint(len(_RULES[sym]))]
    out: list[str] = []
    for tok in prod:
        if tok in _RULES:
            out += gen(tok, rng, depth + 1)
        else:
            out.append(tok)
    return out


def _valid(expr: list[str]) -> bool:
    """Check balanced parens + alternation (toy validator)."""
    depth = 0
    prev_op = True
    for t in expr:
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
            if depth < 0 or prev_op:
                return False
        elif t in "+*":
            if prev_op:
                return False
            prev_op = True
            continue
        prev_op = t == "(" or False if t == "(" else False
        prev_op = t in "(n" and t != "(" or t == "n"
        if t == ")":
            prev_op = False
        if t == "n":
            prev_op = False
        if t == "(":
            prev_op = True
    return depth == 0


def bench_grammar_fuzz(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    valids = 0.0
    trials = 100
    for _ in range(trials):
        expr = gen("E", rng)
        valids += float(_valid(expr))
    return {"synthetic_grammar_valid": valids / trials}
