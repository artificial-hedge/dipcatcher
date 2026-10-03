"""GF(p) arithmetic, primitive elements, and discrete log (wave 281).

Field axioms via brute-force table checks; primitive element generates all
nonzero residues; discrete log inverts exponentiation exactly.
"""

_SEED = 20261231 + 772


def is_primitive(g: int, p: int) -> bool:
    seen = set()
    x = 1
    for _ in range(p - 1):
        x = x * g % p
        seen.add(x)
    return len(seen) == p - 1


def dlog(g: int, h: int, p: int) -> int:
    x = 1
    for k in range(p - 1):
        if x == h:
            return k
        x = x * g % p
    return -1


def _field_axioms(p: int) -> bool:
    for a in range(p):
        for b in range(p):
            for c in range(p):
                if (a + b) % p * c % p != (a * c + b * c) % p:
                    return False
                if (a * b) % p * c % p != a * (b * c) % p:
                    return False
    return all(any(a * x % p == 1 for x in range(1, p)) for a in range(1, p))


def bench_galois_field(seed: int = _SEED) -> dict[str, float]:
    ok = int(_field_axioms(7))
    ok += int(is_primitive(3, 7) and not is_primitive(2, 7))
    ok += int(all(dlog(2, pow(2, k, 11), 11) == k for k in range(10)))
    return {"synthetic_gf_axioms": float(ok == 3)}
