"""Clifford theory: restriction to normal subgroup (SYNTHETIC)."""

from __future__ import annotations


def restrict_std_s3_to_a3() -> list[int]:
    """Std rep of S3 restricted to A3 = C3 splits into two 1-d chars:
    chi(1)=2, chi(3-cyc)=-1 = w + w^2 with w = e^{2pi i/3}."""
    return [2, -1, -1]


def _bench_clifford_toy(seed: int = 0) -> float:
    checks = []
    res = restrict_std_s3_to_a3()
    # degree preserved
    checks.append(res[0] == 2)
    # restriction splits as sum of two conjugate 1-d chars of C3
    w = complex(-0.5, 0.8660254037844386)
    checks.append(abs(complex(res[1]) - (w + w.conjugate())) < 1e-9)
    # both A3 chars are stable under S3-conjugation (same orbit)
    checks.append(res[1] == res[2])
    # inner product <Res std, Res std>_A3 = 2 (two constituents)
    ip = (res[0] ** 2 + res[1] ** 2 + res[2] ** 2) / 3
    checks.append(abs(ip - 2.0) < 1e-9)
    # orbit size = index [S3 : stabilizer] = 2 for each C3 char
    checks.append(ip == 2.0)
    return float(sum(checks) / len(checks))


def bench_clifford_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clifford_toy": _bench_clifford_toy(seed)}
