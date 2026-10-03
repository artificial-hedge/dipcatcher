"""S3 character table: trivial, sign, standard — row orthogonality (SYNTHETIC)."""

from __future__ import annotations

# classes: e=(1), transpositions=(3), 3-cycles=(2)
CLASSES = [("e", 1), ("t", 3), ("c", 2)]
TABLE = {
    "triv": [1, 1, 1],
    "sign": [1, -1, 1],
    "std": [2, 0, -1],
}


def inner_prod(chi1: list[int], chi2: list[int]) -> float:
    """<chi1,chi2> = (1/6) sum_c |c| chi1(c) chi2(c)."""
    num = sum(size * a * b for (_, size), a, b in zip(CLASSES, chi1, chi2, strict=True))
    return num / 6.0


def is_class_func(chi: list[int]) -> bool:
    return len(chi) == len(CLASSES)


def _bench_character_table_s3(seed: int = 0) -> float:
    checks = []
    names = ["triv", "sign", "std"]
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            ip = inner_prod(TABLE[n1], TABLE[n2])
            checks.append(abs(ip - (1.0 if i == j else 0.0)) < 1e-9)
    # std = perm - triv on classes: perm char = [3,1,0], std = [2,0,-1]
    checks.append(TABLE["std"] == [3 - 1, 1 - 1, 0 - 1])
    # sum of squares of dims = |G|
    checks.append(sum(t[0] ** 2 for t in TABLE.values()) == 6)
    checks.append(all(is_class_func(t) for t in TABLE.values()))
    return float(sum(checks) / len(checks))


def bench_character_table_s3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_character_table_s3": _bench_character_table_s3(seed)}
