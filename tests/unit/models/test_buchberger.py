from fractions import Fraction

from quant_fund.models.buchberger import _reduce, _spoly, bench_buchberger, buchberger


def _gens():
    f1 = {(2, 0): Fraction(1), (0, 2): Fraction(1), (0, 0): Fraction(-4)}
    f2 = {(1, 1): Fraction(1), (0, 0): Fraction(-1)}
    return f1, f2


def test_generators_reduce_to_zero_mod_basis():
    """Probe replacing the old float(gcd(12,8)==4) tautology: the
    input generators must reduce to 0 modulo their own Gröbner basis —
    a real invariant the bench now reports."""
    f1, f2 = _gens()
    g = buchberger([f1, f2])
    assert len(_reduce(f1, g)) == 0
    assert len(_reduce(f2, g)) == 0


def test_spoly_leading_terms_cancel():
    f1, f2 = _gens()
    s = _spoly(f1, f2)
    # S-poly kills the LCM leading monomial — no monomial of degree
    # (2,1) or higher survives that isn't reachable from below
    assert (2, 1) not in s
    assert s  # nonzero remainder for this pair


def test_ideal_member_reduces():
    f1, f2 = _gens()
    g = buchberger([f1, f2])
    # x*f2 - y*f1 is in the ideal
    from quant_fund.models.buchberger import _add, _mul_monomial

    probe = _add(
        _mul_monomial(f2, (1, 0), Fraction(1)),
        _mul_monomial(f1, (0, 1), Fraction(1)),
        Fraction(-1),
    )
    assert len(_reduce(probe, g)) == 0


def test_nonmember_stays():
    f1, f2 = _gens()
    g = buchberger([f1, f2])
    assert len(_reduce({(1, 0): Fraction(1), (0, 0): Fraction(100)}, g)) > 0


def test_bench_perfect():
    out = bench_buchberger()
    assert out["synthetic_member_reduces"] == 1.0
    assert out["synthetic_nonmember_stays"] == 1.0
    assert out["synthetic_gens_reduce"] == 1.0
    assert out["synthetic_max_resid"] < 1e-6
