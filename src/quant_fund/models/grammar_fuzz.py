"""Grammar-based fuzzer: generate valid expressions from a CFG (SYNTHETIC)."""

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
    """Balanced parens + operand/operator alternation (toy validator)."""
    depth = 0
    expect_operand = True
    for t in expr:
        if t == "(":
            if not expect_operand:
                return False  # operand followed by '(' has no operator between
            depth += 1
        elif t == ")":
            depth -= 1
            if depth < 0 or expect_operand:
                return False
            expect_operand = False
        elif t in "+*":
            if expect_operand:
                return False
            expect_operand = True
        else:  # operand
            if not expect_operand:
                return False
            expect_operand = False
    return depth == 0 and not expect_operand


def bench_grammar_fuzz(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    valids = 0.0
    trials = 100
    for _ in range(trials):
        expr = gen("E", rng)
        valids += float(_valid(expr))
    out = {"synthetic_grammar_valid": valids / trials}
    # every string a CFG generates is valid by construction — and the
    # validator must reject hand-made invalid strings too
    wrongly_valid = [
        _valid(["n", "+"]),
        _valid(["n", "+", "+", "n"]),
        _valid(["n", ")"]),
        _valid(["(", "n"]),
    ]
    if out["synthetic_grammar_valid"] < 1.0 or any(wrongly_valid) or not _valid(["(", "n", ")"]):
        raise ValueError(f"grammar fuzz off: {out} rej={wrongly_valid}")
    return out
