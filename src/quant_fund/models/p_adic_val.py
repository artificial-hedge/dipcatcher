"""p-adic valuation, ultrametric norm, Hensel-lift verification (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations


def v_p(n: int, p: int) -> int:
    if n == 0:
        return 999  # infinity
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def p_norm(n: int, p: int) -> float:
    if n == 0:
        return 0.0
    return float(p) ** (-v_p(n, p))


def ultrametric(a: int, b: int, c: int, p: int) -> bool:
    """|a-c| <= max(|a-b|,|b-c|) under p-adic norm (as exponents)."""
    va, vb, vc = v_p(a - b, p), v_p(b - c, p), v_p(a - c, p)
    return vc >= min(va, vb)


def hensel_lift(f, fp, p: int, x0: int, iters: int = 5) -> int:
    """x_{k+1} = x_k - f(x_k)/f'(x_k) mod p^{k+1}; needs f'(x0) unit mod p."""
    x = x0 % p
    mod = p
    for _ in range(iters):
        fx = f(x) % (mod * p)
        fpx = fp(x) % p
        x = (x - fx * pow(fpx, -1, p)) % (mod * p)
        mod *= p
    return x


def _bench_p_adic_val(seed: int = 0) -> float:
    checks = []
    checks.append(v_p(40, 2) == 3)
    checks.append(v_p(7, 7) == 1)
    checks.append(v_p(1, 5) == 0)
    checks.append(abs(p_norm(8, 2) - 0.125) < 1e-9)
    checks.append(ultrametric(0, 4, 12, 2))
    checks.append(ultrametric(3, 6, 9, 3))
    # x^2 = 2 mod 7: x0=3 root (9=2 mod7); lift to mod 49: 10^2=100=2 mod49? 100-98=2 yes
    root = hensel_lift(lambda x: x * x - 2, lambda x: 2 * x, 7, 3, iters=1)
    checks.append((root * root - 2) % 49 == 0)
    return float(sum(checks) / len(checks))


def bench_p_adic_val(seed: int = 0) -> dict[str, float]:
    return {"synthetic_p_adic_val": _bench_p_adic_val(seed)}
