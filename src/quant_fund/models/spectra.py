"""Stable homotopy groups / spectra bookkeeping (SYNTHETIC)."""

from __future__ import annotations

# Known stable stems pi_n^S for small n (Serrano/standard tables):
# pi_0^S = Z, pi_1^S = Z/2, pi_2^S = Z/2, pi_3^S = Z/24, pi_4^S = 0,
# pi_5^S = 0, pi_6^S = Z/2, pi_7^S = Z/240
_STABLE_STEMS = {0: "Z", 1: "Z/2", 2: "Z/2", 3: "Z/24", 4: "0", 5: "0", 6: "Z/2", 7: "Z/240"}


def stable_stem(n: int) -> str:
    return _STABLE_STEMS[n]


def unstable_pi(k: int, n: int) -> str:
    """Toy table of pi_{n+k}(S^n) in low degrees."""
    table = {
        (0, 1): "Z",
        (0, 2): "Z",
        (1, 2): "Z",  # pi_3(S2) = Z (Hopf)
        (0, 3): "Z",
        (1, 3): "Z/2",  # pi_4(S3) = Z/2
        (2, 3): "Z/2",  # pi_5(S3) = Z/2
        (0, 4): "Z",
        (1, 4): "Z/2",  # pi_5(S4)
        (2, 4): "Z/2",  # pi_6(S4)
        (3, 4): "Z x Z/12",  # pi_7(S4)
    }
    return table.get((k, n), "?")


def _bench_spectra(seed: int = 0) -> float:
    checks = []
    # stable stems known values
    checks.append(stable_stem(0) == "Z")
    checks.append(stable_stem(1) == "Z/2")
    checks.append(stable_stem(3) == "Z/24")
    checks.append(stable_stem(4) == "0")
    # stabilization: pi_{n+1}(S^n) -> pi_1^S = Z/2 for n >= 3
    checks.append(unstable_pi(1, 3) == "Z/2" == stable_stem(1))
    checks.append(unstable_pi(1, 4) == stable_stem(1))
    # pi_3(S2) = Z (Hopf invariant), becomes torsion after suspension
    checks.append(unstable_pi(1, 2) == "Z")
    checks.append(unstable_pi(1, 3) == "Z/2")
    # pi_2^S = Z/2 equals pi_4(S^3)? pi_{n+2}(S^n) for n>=4: pi_6(S4)=Z/2
    checks.append(unstable_pi(2, 4) == stable_stem(2))
    return float(sum(checks) / len(checks))


def bench_spectra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectra": _bench_spectra(seed)}
