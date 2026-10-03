from quant_fund.models.block_low_rank import (
    bench_block_low_rank,
)
from quant_fund.models.h_matrix import (
    bench_h_matrix,
)
from quant_fund.models.hss_matrix import (
    bench_hss_matrix,
)
from quant_fund.models.kronecker_approx import (
    bench_kronecker_approx,
)
from quant_fund.models.low_rank_svd import (
    bench_low_rank_svd,
)
from quant_fund.models.randomized_nystrom import (
    bench_randomized_nystrom,
)


def test_low_rank_svd():
    assert bench_low_rank_svd()["synthetic_low_rank_svd"] == 1.0


def test_h_matrix():
    assert bench_h_matrix()["synthetic_h_matrix"] == 1.0


def test_hss_matrix():
    assert bench_hss_matrix()["synthetic_hss_matrix"] == 1.0


def test_randomized_nystrom():
    assert bench_randomized_nystrom()["synthetic_randomized_nystrom"] == 1.0


def test_block_low_rank():
    assert bench_block_low_rank()["synthetic_block_low_rank"] == 1.0


def test_kronecker_approx():
    assert bench_kronecker_approx()["synthetic_kronecker_approx"] == 1.0
