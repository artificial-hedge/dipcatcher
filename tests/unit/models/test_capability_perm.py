import pytest

from quant_fund.models.capability_perm import PermError, bench_capability_perm, run


def test_join_requires_paired_halves() -> None:
    # halves born of DIFFERENT splits must not recombine — the old model
    # accepted any two "U_split" entries and fused unrelated parents.
    with pytest.raises(PermError):
        run(
            [
                ("alloc", "x"),
                ("split", "x", "a", "b"),
                ("alloc", "z"),
                ("split", "z", "c", "d"),
                ("join", "a", "c", "w"),
            ]
        )


def test_join_paired_halves_ok() -> None:
    st = run([("alloc", "x"), ("split", "x", "a", "b"), ("join", "a", "b", "x")])
    assert st["x"] == "U"


def test_join_swapped_order_ok() -> None:
    st = run([("alloc", "x"), ("split", "x", "a", "b"), ("join", "b", "a", "x")])
    assert st["x"] == "U"


def test_drop_ro_rejects_owner() -> None:
    # drop_ro must only retire readonly views; deleting the unique owner
    # would leave dangling R aliases and violates the capability model.
    with pytest.raises(PermError):
        run([("alloc", "x"), ("alias_ro", "x", "y"), ("drop_ro", "x")])
    with pytest.raises(PermError):
        run([("alloc", "x"), ("drop_ro", "x")])


def test_drop_ro_removes_view() -> None:
    st = run([("alloc", "x"), ("alias_ro", "x", "y"), ("drop_ro", "y")])
    assert "y" not in st
    assert st["x"] == "U"


def test_split_halves_not_writable() -> None:
    with pytest.raises(PermError):
        run([("alloc", "x"), ("split", "x", "a", "b"), ("write", "a")])


def test_bench_passes() -> None:
    assert bench_capability_perm()["synthetic_capability_perm"] == 1.0
