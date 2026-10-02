from quant_fund.models.exact_cover import (
    algorithm_x,
    bench_exact_cover,
    exact_cover_matrix,
)


def test_simple_cover():
    rows = [[0, 1], [2, 3], [0, 2], [1, 3]]
    mat = exact_cover_matrix(rows, 4)
    sols = algorithm_x(mat, limit=10)
    assert len(sols) == 2  # {0,1} or {2,3}
    assert sorted(sols[0]) in ([0, 1], [2, 3])


def test_no_solution():
    mat = exact_cover_matrix([[0, 1], [0]], 3)
    assert algorithm_x(mat) == []


def test_unique_solution():
    rows = [[0, 2], [1], [0, 1], [2]]
    mat = exact_cover_matrix(rows, 3)
    sols = algorithm_x(mat, limit=10)
    assert len(sols) == 2  # {0,1} and {2,3}
    for s in sols:
        assert sorted(s) in ([0, 1], [2, 3])


def test_bench_keys():
    out = bench_exact_cover()
    assert out["synthetic_cover_valid"] == 1.0
    assert out["synthetic_nqueens"] == 2.0
