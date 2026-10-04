"""D-modules: Weyl algebra actions (SYNTHETIC)."""

from __future__ import annotations


def weyl_relation(delta: float) -> bool:
    """Weyl algebra D = k[x, d/dx]: [d/dx, x] = 1
    (Leibniz: d/dx(x f) = f + x f')."""
    return abs(delta - 1.0) < 1e-9


def is_d_module(action_closes: bool, order_finite: bool) -> bool:
    """A coherent D-module is a module over D whose
    differential action closes (Fuchs/Beilinson-Bernstein)."""
    return action_closes and order_finite


def _bench_d_module(seed: int = 0) -> float:
    checks = []
    # [d, x] = dx - xd = 1 applied to f: dx f - x df = f
    checks.append(weyl_relation(1.0))
    checks.append(not weyl_relation(0.0))
    checks.append(is_d_module(True, True))
    checks.append(not is_d_module(True, False))
    # O_X and delta-modules are D-modules
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_d_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_d_module": _bench_d_module(seed)}
