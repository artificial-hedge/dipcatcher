"""Elias gamma/delta universal integer coding (wave 285).

gamma(n) = floor(log2 n) zeros + binary n. Prefix-free: decode roundtrips
and codeword length is monotone-ish (nondecreasing on powers of two gaps).
"""

_SEED = 20261231 + 798


def gamma_encode(n: int) -> str:
    b = bin(n)[2:]
    return "0" * (len(b) - 1) + b


def gamma_decode(s: str) -> tuple[int, int]:
    zeros = 0
    while s[zeros] == "0":
        zeros += 1
    return int("1" + s[zeros + 1 : zeros + 1 + zeros], 2), zeros + 1 + zeros


def delta_encode(n: int) -> str:
    b = bin(n)[2:]
    return gamma_encode(len(b)) + b[1:]


def delta_decode(s: str) -> tuple[int, int]:
    ln, used = gamma_decode(s)
    body = s[used : used + ln - 1]
    return int("1" + body, 2), used + ln - 1


def bench_elias_gamma(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    for n in range(1, 65):
        dec, used = gamma_decode(gamma_encode(n))
        ok += int(dec == n and used == len(gamma_encode(n)))
    # prefix-free check: no codeword is a prefix of another
    codes = [gamma_encode(n) for n in range(1, 65)]
    ok += int(
        not any(
            codes[i] != codes[j] and codes[j].startswith(codes[i])
            for i in range(64)
            for j in range(64)
        )
    )
    # delta roundtrip
    ok += int(all(delta_decode(delta_encode(n)) == (n, len(delta_encode(n))) for n in range(1, 33)))
    return {"synthetic_elias": float(ok >= 66)}
