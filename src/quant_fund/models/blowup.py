"""Blow-up of the plane at a point and strict transforms (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def strict_transform_x_chart(
    coeffs: dict[tuple[int, int], Fraction],
) -> dict[str, Fraction]:
    """Pull back a curve f(x,y)=0 to the x-chart of the blow-up: y = x*u.

    The exceptional divisor x = 0 factors out f(x, x*u) = x^e * g(x, u);
    return g's coefficients (as a polynomial in x, u) keyed 'x^i u^j'.
    """
    # represent terms as (i,j) -> coeff for x^i y^j; substitute y = x*u:
    # x^i (x u)^j = x^{i+j} u^j ; collect on (i+j, j)
    out: dict[tuple[int, int], Fraction] = {}
    e_min: int | None = None
    for (i, j), c in coeffs.items():
        key = (i + j, j)
        out[key] = out.get(key, Fraction(0)) + c
        if e_min is None or i + j < e_min:
            e_min = i + j
    if not (e_min is not None):
        raise ValueError("e_min is not None")
    # divide by x^e_min
    g: dict[tuple[int, int], Fraction] = {}
    for (i, j), c in out.items():
        g[(i - e_min, j)] = c
    return {f"x^{i} u^{j}": c for (i, j), c in g.items()}


def eval_g(g: dict[str, Fraction], x: Fraction, u: Fraction) -> Fraction:
    total = Fraction(0)
    for k, c in g.items():
        i = int(k.split("x^")[1].split(" ")[0])
        j = int(k.split("u^")[1])
        total += c * x**i * u**j
    return total


def _bench_blowup(seed: int = 0) -> float:
    checks = []
    # nodal cubic y^2 = x^3 + x^2 -> f = -y^2 + x^3 + x^2
    f: dict[tuple[int, int], Fraction] = {}
    f[(3, 0)] = Fraction(1)
    f[(2, 0)] = Fraction(1)
    f[(0, 2)] = Fraction(-1)
    g = strict_transform_x_chart(f)
    # g should be x + 1 - u^2 : smooth conic u^2 = x + 1
    checks.append(g.get("x^1 u^0") == 1 and g.get("x^0 u^0") == 1)
    checks.append(g.get("x^0 u^2") == -1)
    checks.append(len(g) == 3)
    # smoothness of the strict transform at its intersections with the
    # exceptional divisor (x=0, u = +-1): dg/dx = 1 != 0 -> smooth
    checks.append(eval_g(g, Fraction(0), Fraction(1)) == 0)
    checks.append(eval_g(g, Fraction(0), Fraction(-1)) == 0)
    # cusp y^2 = x^3: strict transform x^3 - (xu)^2 = x^2(x - u^2) -> u^2 = x smooth
    f2: dict[tuple[int, int], Fraction] = {(3, 0): Fraction(1), (0, 2): Fraction(-1)}
    g2 = strict_transform_x_chart(f2)
    checks.append(g2.get("x^1 u^0") == 1 and g2.get("x^0 u^2") == -1 and len(g2) == 2)
    return float(sum(checks) / len(checks))


def bench_blowup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_blowup": _bench_blowup(seed)}
