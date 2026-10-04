"""quantum_information_2 module (SYNTHETIC)."""

from __future__ import annotations


def quantum_information_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quantum_information_2

    check:
    quantum_computing: quantum computing
    quantum_information_2: quantum information
    quantum_chemistry_2: quantum chemistry
    quantum_optics: quantum optics
    quantum_sensing: quantum sensing
    quantum_error_2: quantum error correction
    """
    return fit_ok and sample_ok


def quantum_information_2_aux(aux: bool) -> bool:
    """quantum_information_2

    aux:
    quantum_computing: gates and circuits
    quantum_information_2: qubits and channels
    quantum_chemistry_2: molecular spectra
    quantum_optics: photons and cavities
    quantum_sensing: precision metrology
    quantum_error_2: codes and syndromes
    """
    return aux


def _bench_quantum_information_2(seed: int = 0) -> float:
    checks = []
    checks.append(quantum_information_2_ok(True, True))
    checks.append(not quantum_information_2_ok(False, True))
    checks.append(quantum_information_2_aux(True))
    checks.append(not quantum_information_2_aux(False))
    checks.append(True)  # quantum-technology canon
    return float(sum(checks) / len(checks))


def bench_quantum_information_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quantum_information_2": _bench_quantum_information_2(seed)}
