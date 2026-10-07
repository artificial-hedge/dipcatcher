import pytest

from quant_fund.models.bidirectional_tc import (
    BOOL,
    NAT,
    TypeError_,
    arrow,
    bench_bidirectional_tc,
    check,
    infer,
    prod,
)


def test_infer_basic():
    assert infer(("lit", 5), {}) == NAT
    assert infer(("lit", True), {}) == BOOL
    assert infer(("pair", ("lit", 1), ("lit", False)), {}) == prod(NAT, BOOL)


def test_check_lam_rejects_mismatched_annotation():
    # check-mode must not discard the lambda's declared domain:
    # the declared BOOL param cannot satisfy an expected NAT->BOOL.
    with pytest.raises(TypeError_):
        check(("lam", "x", BOOL, ("lit", True)), arrow(NAT, BOOL), {})
    # same expression infers to a different arrow than the one checked
    assert infer(("lam", "x", BOOL, ("lit", True)), {}) == arrow(BOOL, BOOL)


def test_check_lam_accepts_matching_annotation():
    check(("lam", "x", NAT, ("lit", True)), arrow(NAT, BOOL), {})


def test_check_pair_and_if():
    check(("pair", ("lit", 1), ("lit", True)), prod(NAT, BOOL), {})
    check(("if", ("lit", False), ("lit", 1), ("lit", 2)), NAT, {})
    with pytest.raises(TypeError_):
        check(("pair", ("lit", 1), ("lit", True)), prod(BOOL, NAT), {})


def test_bench_contract():
    assert bench_bidirectional_tc() == {"synthetic_bidirectional_tc": 1.0}
