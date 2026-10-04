"""Conditional expectation + tower property on finite spaces (SYNTHETIC)."""

from __future__ import annotations

Prob = dict[tuple[int, int], float]


def marginal_x(joint: Prob) -> dict[int, float]:
    out: dict[int, float] = {}
    for (x, _), v in joint.items():
        out[x] = out.get(x, 0.0) + v
    return out


def cond_exp_y(joint: Prob, x: int) -> float:
    """E[Y | X=x]."""
    num = sum(y * v for (xx, y), v in joint.items() if xx == x)
    den = sum(v for (xx, y), v in joint.items() if xx == x)
    return num / den


def exp_x(joint: Prob) -> float:
    return sum(x * v for (x, _), v in joint.items())


def _bench_conditional_expect(seed: int = 0) -> float:
    checks = []
    joint = {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4}
    # E[Y|X=0] = (0*0.1 + 1*0.2)/0.3 = 2/3
    checks.append(abs(cond_exp_y(joint, 0) - 2.0 / 3.0) < 1e-9)
    # tower property: E[E[Y|X]] = E[Y]
    mx = marginal_x(joint)
    tower = sum(cond_exp_y(joint, x) * mx[x] for x in mx)
    ey = sum(y * v for (xx, y), v in joint.items())
    checks.append(abs(tower - ey) < 1e-9)
    checks.append(abs(ey - 0.6) < 1e-9)
    # E[X] = 0.7
    checks.append(abs(exp_x(joint) - 0.7) < 1e-9)
    # conditional expectation is measurable to sigma(X): const per x
    checks.append(abs(cond_exp_y(joint, 1) - 0.4 / 0.7) < 1e-9)
    # independence case: E[Y|X] = E[Y]
    indep = {(x, y): 0.25 for x in (0, 1) for y in (0, 1)}
    checks.append(abs(cond_exp_y(indep, 0) - 0.5) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_conditional_expect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conditional_expect": _bench_conditional_expect(seed)}
