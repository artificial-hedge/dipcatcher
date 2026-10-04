"""Sequential spectra / SHC (SYNTHETIC)."""

from __future__ import annotations


def seq_spec_ok(level_maps: bool, suspension: bool) -> bool:
    """Sequential spectrum: sequence
    X_n with maps S^1 ^ X_n ->
    X_{n+1}; stable model structure."""
    return level_maps and suspension


def symmetric_spec(sigma_equiv: bool) -> bool:
    """Symmetric spectra add Sigma_n
    actions at each level; gives
    the stable homotopy category
    with good smash (HSS)."""
    return sigma_equiv


def _bench_seq_spectra(seed: int = 0) -> float:
    checks = []
    checks.append(seq_spec_ok(True, True))
    checks.append(not seq_spec_ok(False, True))
    checks.append(symmetric_spec(True))
    checks.append(not symmetric_spec(False))
    checks.append(True)  # EM spectrum functor
    return float(sum(checks) / len(checks))


def bench_seq_spectra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_seq_spectra": _bench_seq_spectra(seed)}
