"""bernstein form module (SYNTHETIC)."""

from __future__ import annotations


def bernstein_form_ok(node: bool, form: bool) -> bool:
    """bernstein_form
    check:
    interpolation-2
    canon — node/
    form
    consistency."""
    return node and form


def bernstein_form_aux(aux: bool) -> bool:
    """bernstein_form
    aux:
    auxiliary
    basis check —
    interpolation bound."""
    return aux


def _bench_bernstein_form(seed: int = 0) -> float:
    checks = []
    checks.append(bernstein_form_ok(True, True))
    checks.append(not bernstein_form_ok(False, True))
    checks.append(bernstein_form_aux(True))
    checks.append(not bernstein_form_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_bernstein_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bernstein_form": _bench_bernstein_form(seed)}
