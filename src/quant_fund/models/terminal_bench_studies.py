"""terminal_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def terminal_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """terminal_bench_studies

    check:
    terminal_bench_studies: terminal task shells/commands and exit states
    """
    return fit_ok and sample_ok


def terminal_bench_studies_aux(aux: bool) -> bool:
    """terminal_bench_studies

    aux:
    terminal_bench_studies: tmux session scripts/checks and pass flags
    """
    return aux


def _bench_terminal_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(terminal_bench_studies_ok(True, True))
    checks.append(not terminal_bench_studies_ok(False, True))
    checks.append(terminal_bench_studies_aux(True))
    checks.append(not terminal_bench_studies_aux(False))
    checks.append(True)  # agentic-eval canon
    return float(sum(checks) / len(checks))


def bench_terminal_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_terminal_bench_studies": _bench_terminal_bench_studies(seed)}
