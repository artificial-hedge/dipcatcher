"""secp256k1 scalar multiplication — real curve, affine coordinates (SYNTHETIC).

Implements point add/double over F_p and double-and-add scalar mult.
Bench: k*G on-curve, (k*G).x matches the published k=2 doubling vector,
n*G = O (group order), and scalar-mult agreement with repeated addition.
"""

from quant_fund.models._crypto_synth import SECP_GX, SECP_GY, SECP_N, SECP_P

P2 = SECP_P


def _add(P: tuple[int, int] | None, Q: tuple[int, int] | None) -> tuple[int, int] | None:
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % P2 == 0:
        return None
    if P == Q:
        lam = (3 * x1 * x1) * pow(2 * y1, P2 - 2, P2) % P2
    else:
        lam = (y2 - y1) * pow(x2 - x1, P2 - 2, P2) % P2
    x3 = (lam * lam - x1 - x2) % P2
    y3 = (lam * (x1 - x3) - y1) % P2
    return (x3, y3)


def _mul(k: int, P: tuple[int, int]) -> tuple[int, int] | None:
    r: tuple[int, int] | None = None
    q: tuple[int, int] | None = P
    while k:
        if k & 1:
            r = _add(r, q)
        q = _add(q, q)
        k >>= 1
    return r


def _on_curve(P: tuple[int, int]) -> bool:
    x, y = P
    return (y * y - x * x * x - 7) % P2 == 0


G = (SECP_GX, SECP_GY)


def bench_ecc_secp256k1(seed: int = 4711) -> dict[str, float]:
    del seed
    two_g = _add(G, G)
    # published 2G.x for secp256k1
    two_g_x_ref = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
    k = 12345
    kg = _mul(k, G)
    ng = _mul(SECP_N, G)
    # n*G should be the point at infinity (None)
    rep: tuple[int, int] | None = G
    for _ in range(5):
        rep = _add(rep, G)  # rep = 6G
    six_g = _mul(6, G)
    return {
        "synthetic_ecc_oncurve_g": float(_on_curve(G)),
        "synthetic_ecc_oncurve_kg": float(kg is not None and _on_curve(kg)),
        "synthetic_ecc_2g_x_match": float(two_g is not None and two_g[0] == two_g_x_ref),
        "synthetic_ecc_order": float(ng is None),
        "synthetic_ecc_add_mul": float(rep == six_g),
    }
