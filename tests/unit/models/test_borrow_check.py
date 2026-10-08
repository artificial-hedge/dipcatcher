from quant_fund.models.borrow_check import bench_borrow_check, check


def test_use_conflicts_with_live_mut_loan():
    """Probe: a direct use of x while a mut loan on x is live is an
    error — the docstring's 'mut borrows conflict with any overlapping
    loan or use of the same place'."""
    errs = check(
        [
            ("borrow", "x", "mut", "L1"),
            ("use", "x"),
            ("loan_use", "L1"),
        ]
    )
    assert len(errs) == 1 and "use" in errs[0]


def test_use_ok_with_shared_loan():
    assert (
        check(
            [
                ("borrow", "x", "shared", "L1"),
                ("use", "x"),
                ("loan_use", "L1"),
            ]
        )
        == []
    )


def test_use_after_mut_loan_dies_is_ok():
    assert (
        check(
            [
                ("borrow", "x", "mut", "L1"),
                ("loan_use", "L1"),
                ("use", "x"),
            ]
        )
        == []
    )


def test_use_of_other_place_ok():
    assert (
        check(
            [
                ("borrow", "x", "mut", "L1"),
                ("use", "y"),
                ("loan_use", "L1"),
            ]
        )
        == []
    )


def test_bench_perfect():
    assert bench_borrow_check()["synthetic_borrow_check"] == 1.0
