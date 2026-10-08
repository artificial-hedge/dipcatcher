"""OSR deoptimization frame reconstruction (SYNTHETIC).

An optimized frame keeps a deopt map: virtual objects (allocations
eliminated by escape analysis) and their field values, plus the
baseline pc. Deopt materializes the baseline frame exactly.

Baseline program ops: (op, dst, a, b) with op in {add, sub, mul};
operand b is an int literal or a local name.
"""

from __future__ import annotations

_SEED = 20261231 + 898

Baseline = list[tuple[str, str, str, object]]

_OPS = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y}


def run_baseline(prog: Baseline, start: int, env: dict[str, int]) -> dict[str, int]:
    """Interpret baseline from instruction `start` with given locals."""
    out = dict(env)
    for op, dst, a, b in prog[start:]:
        bv = out[b] if isinstance(b, str) else b
        out[dst] = _OPS[op](out[a], bv)
    return out


class OptimizedFrame:
    """Optimized frame with virtual objects and a deopt continuation."""

    def __init__(self, prog: Baseline, pc: int) -> None:
        self.prog = prog
        self.pc = pc  # baseline resume point
        self.locals: dict[str, int] = {}
        self.virtual: dict[str, dict[str, int]] = {}  # obj -> fields

    def deopt(self) -> dict[str, int]:
        """Materialize baseline state: virtual objects become flat locals."""
        env = dict(self.locals)
        for name, fields in self.virtual.items():
            for fname, val in fields.items():
                env[f"{name}_{fname}"] = val
        return run_baseline(self.prog, self.pc, env)


def bench_osr_deopt(seed: int = _SEED) -> dict[str, float]:
    prog: Baseline = [
        ("add", "a", "b", 1),
        ("mul", "c", "a", 2),
        ("sub", "d", "c", "p_x"),
        ("add", "d2", "d", "p_y"),
        ("mul", "e", "d2", 3),
    ]
    oracle = run_baseline(prog, 0, {"b": 4, "p_x": 10, "p_y": 20})
    # optimized run: pc=0..1 done (a=5,c=10); object p virtual
    fr = OptimizedFrame(prog, 2)
    fr.locals = {"b": 4, "a": 5, "c": 10}
    fr.virtual["p"] = {"x": 10, "y": 20}
    env = fr.deopt()
    score = 0.0
    score += 1.0 if env == oracle else 0.0
    # deopt at pc=0: cold-start reconstruction also exact
    fr0 = OptimizedFrame(prog, 0)
    fr0.locals = {"b": 4}
    fr0.virtual["p"] = {"x": 10, "y": 20}
    score += 1.0 if fr0.deopt() == oracle else 0.0
    # missing field materialization detected
    fr_bad = OptimizedFrame(prog, 2)
    fr_bad.locals = {"b": 4, "a": 5, "c": 10}
    fr_bad.virtual["p"] = {"x": 10}
    try:
        fr_bad.deopt()
        bad_caught = False
    except KeyError:
        bad_caught = True
    score += 1.0 if bad_caught else 0.0
    # materialized local set matches oracle coverage
    score += 1.0 if set(env) == set(oracle) else 0.0
    return {"synthetic_osr_deopt": score / 4.0}
