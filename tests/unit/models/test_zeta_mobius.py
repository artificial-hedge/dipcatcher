"""Probes for zeta_mobius — incidence-algebra invariants."""

from quant_fund.models.zeta_mobius import mobius


def test_mu_zeta_convolution_is_identity() -> None:
    """sum_y mu(x,y) for x<=y<=z == 0 for x<z, 1 for x==z (mu*zeta = delta)."""

    def leq(a: int, b: int) -> bool:
        return a <= b

    elems = tuple(range(6))
    mu = mobius(leq, elems)
    for x in elems:
        for z in elems:
            s = sum(mu[(x, y)] for y in elems if leq(x, y) and leq(y, z))
            assert s == (1 if x == z else 0)
