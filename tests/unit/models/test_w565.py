from quant_fund.models.bhargava_lic import bench_bhargava_lic
from quant_fund.models.cohen_lenstra import bench_cohen_lenstra
from quant_fund.models.elliptic_rank import bench_elliptic_rank
from quant_fund.models.malle_conj import bench_malle_conj
from quant_fund.models.prime_gaps import bench_prime_gaps
from quant_fund.models.zhang_maynard import bench_zhang_maynard


def test_bhargava_lic():
    assert bench_bhargava_lic()["synthetic_bhargava_lic"] == 1.0


def test_cohen_lenstra():
    assert bench_cohen_lenstra()["synthetic_cohen_lenstra"] == 1.0


def test_elliptic_rank():
    assert bench_elliptic_rank()["synthetic_elliptic_rank"] == 1.0


def test_malle_conj():
    assert bench_malle_conj()["synthetic_malle_conj"] == 1.0


def test_prime_gaps():
    assert bench_prime_gaps()["synthetic_prime_gaps"] == 1.0


def test_zhang_maynard():
    assert bench_zhang_maynard()["synthetic_zhang_maynard"] == 1.0
