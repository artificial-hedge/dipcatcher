from quant_fund.models.distality import bench_distality
from quant_fund.models.dp_rank import bench_dp_rank
from quant_fund.models.forking_seq import bench_forking_seq
from quant_fund.models.honest_def import bench_honest_def
from quant_fund.models.nip_formula import bench_nip_formula
from quant_fund.models.uniform_def import bench_uniform_def


def test_dp_rank():
    assert bench_dp_rank()["synthetic_dp_rank"] == 1.0


def test_forking_seq():
    assert bench_forking_seq()["synthetic_forking_seq"] == 1.0


def test_honest_def():
    assert bench_honest_def()["synthetic_honest_def"] == 1.0


def test_uniform_def():
    assert bench_uniform_def()["synthetic_uniform_def"] == 1.0


def test_distality():
    assert bench_distality()["synthetic_distality"] == 1.0


def test_nip_formula():
    assert bench_nip_formula()["synthetic_nip_formula"] == 1.0
