"""Tanimoto similarity (wave 290) (SYNTHETIC).

J(A,B) = |A∩B| / |A∪B| on fingerprint sets; verified against brute-force
Jaccard plus known self/extreme values, and identity A~A=1.
"""

_SEED = 20261231 + 826


def tanimoto(a: frozenset[int], b: frozenset[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def bench_tanimoto(seed: int = _SEED) -> dict[str, float]:
    a, b = frozenset({1, 2, 3}), frozenset({2, 3, 4})
    ok = int(abs(tanimoto(a, b) - 0.5) < 1e-12)
    ok += int(tanimoto(a, a) == 1.0)
    ok += int(tanimoto(a, frozenset({9, 10})) == 0.0)
    # ethanol vs methanol share the -OH env more than ethanol vs cyclopropane
    from quant_fund.models.morgan_fp import morgan
    from quant_fund.models.smiles_parse import parse

    fa = morgan(*parse("CCO"))
    fb = morgan(*parse("CO"))
    fc = morgan(*parse("C1CC1"))
    ok += int(tanimoto(fa, fb) > tanimoto(fa, fc))
    return {"synthetic_tanimoto": float(ok == 4)}
