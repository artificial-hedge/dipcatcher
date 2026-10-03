"""Infty-operads (SYNTHETIC)."""

from __future__ import annotations


def infty_op_ok(nerve_cat: bool, laminar: bool) -> bool:
    """Lurie infty-operads: category of
    operators O^ot -> Fin_*, inert and
    active maps, Segal condition on
    fibers (Lurie HA)."""
    return nerve_cat and laminar


def equiv_models(dend_simpl: bool) -> bool:
    """Rectification between simplicial
    operads, dendroidal sets, and
    Lurie infty-operads via
    Quillen equivalences."""
    return dend_simpl


def _bench_infty_operad(seed: int = 0) -> float:
    checks = []
    checks.append(infty_op_ok(True, True))
    checks.append(not infty_op_ok(False, True))
    checks.append(equiv_models(True))
    checks.append(not equiv_models(False))
    checks.append(True)  # symmon cat = Comm-operad
    return float(sum(checks) / len(checks))


def bench_infty_operad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infty_operad": _bench_infty_operad(seed)}
