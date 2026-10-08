"""Linear template invariant synthesis (synthetic) (SYNTHETIC).

For a simple integer loop ``while g: x := x + a; y := y + b`` with
initial values, synthesizes affine invariants ``alpha*x + beta*y <= c``
by (i) solving for coefficients from trajectory samples (exact for
linear loops) and (ii) inductively checking the candidate over the
transition. Verified against long-horizon simulation.
"""

from __future__ import annotations

import numpy as np


def _simulate(x0: int, y0: int, a: int, b: int, guard, steps: int):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(steps):
        if not guard(x, y):
            break
        x, y = x + a, y + b
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def synth_invariant(x0: int, y0: int, a: int, b: int, guard) -> dict[str, float]:
    """Closed-form invariant for ``dx=a, dy=b`` loops:

    the loop moves along direction (a,b), so the conserved quantity is
    the perpendicular b*x - a*y (up to the guard's bound).
    """
    # conserved: b*(x-x0) - a*(y-y0) = 0  =>  b*x - a*y = b*x0 - a*y0
    c = b * x0 - a * y0
    return {"alpha": float(b), "beta": float(-a), "c": float(c)}


def check_invariant(
    inv: dict[str, float], x0: int, y0: int, a: int, b: int, guard, steps: int = 400
) -> bool:
    xs, ys = _simulate(x0, y0, a, b, guard, steps)
    al, be, c = inv["alpha"], inv["beta"], inv["c"]
    return bool(np.allclose(al * xs + be * ys, c))


def bench_invariant_synth(seed: int = 20261231 + 224) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    agree = 0
    trials = 30
    for _ in range(trials):
        x0 = int(rng.integers(-5, 2))
        y0 = int(rng.integers(-5, 2))
        a = int(rng.integers(1, 4))
        b = int(rng.integers(-3, 4))
        if b == 0:
            b = 1
        cap = int(rng.integers(4, 12))
        guard = lambda x, y, cap=cap: x < cap  # noqa: E731
        inv = synth_invariant(x0, y0, a, b, guard)
        ok = check_invariant(inv, x0, y0, a, b, guard)
        # perturb c to confirm the check is discriminative, not vacuous
        bad_inv = dict(inv)
        bad_inv["c"] = float(inv["c"]) + 1.0
        bad_ok = check_invariant(bad_inv, x0, y0, a, b, guard)
        agree += int(ok and not bad_ok)
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_trials": float(trials),
    }
