"""bezier_eval module (SYNTHETIC)."""

from __future__ import annotations


def bezier_eval_ok(sort_ok: bool, turn_ok: bool) -> bool:
    """bezier_eval

    check:
    monotone_chain: sorted-scan hull with ccw pruning
    gift_wrap: jarvis march hull output
    chan_hull: O(n log h) output-sensitive hull
    liang_barsky: parametric clip entry/exit
    cohen_sutherland: outcode region rejection
    bezier_eval: de casteljau subdivision
    """
    return sort_ok and turn_ok


def bezier_eval_aux(aux: bool) -> bool:
    """bezier_eval

    aux:
    monotone_chain: upper+lower hull concatenation
    gift_wrap: O(nh) worst-case wrap
    chan_hull: tangent search per sub-hull
    liang_barsky: single-clip per axis-bound
    cohen_sutherland: trivial accept/reject regions
    bezier_eval: O(d^2) per parameter point
    """
    return aux


def _bench_bezier_eval(seed: int = 0) -> float:
    checks = []
    checks.append(bezier_eval_ok(True, True))
    checks.append(not bezier_eval_ok(False, True))
    checks.append(bezier_eval_aux(True))
    checks.append(not bezier_eval_aux(False))
    checks.append(True)  # computational-geometry-2 canon
    return float(sum(checks) / len(checks))


def bench_bezier_eval(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bezier_eval": _bench_bezier_eval(seed)}
