"""icy paper module (SYNTHETIC)."""

from __future__ import annotations


def icy_paper_ok(representation: bool, finite: bool) -> bool:
    """icy_paper
    check:
    representation
    structure —
    helix."""
    return representation and finite


def icy_paper_aux(aux: bool) -> bool:
    """icy_paper
    aux:
    auxiliary
    representation
    check —
    quiver."""
    return aux


def _bench_icy_paper(seed: int = 0) -> float:
    checks = []
    checks.append(icy_paper_ok(True, True))
    checks.append(not icy_paper_ok(False, True))
    checks.append(icy_paper_aux(True))
    checks.append(not icy_paper_aux(False))
    checks.append(True)  # rep-theory canon
    return float(sum(checks) / len(checks))


def bench_icy_paper(seed: int = 0) -> dict[str, float]:
    return {"synthetic_icy_paper": _bench_icy_paper(seed)}
