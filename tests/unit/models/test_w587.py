from quant_fund.models.abundance_conj import (
    bench_abundance_conj,
)
from quant_fund.models.bdd_fano import bench_bdd_fano
from quant_fund.models.canonical_sing2 import (
    bench_canonical_sing2,
)
from quant_fund.models.klt_mmp import bench_klt_mmp
from quant_fund.models.mmp_flip import bench_mmp_flip
from quant_fund.models.terminal_sing import (
    bench_terminal_sing,
)


def test_terminal_sing():
    assert bench_terminal_sing()["synthetic_terminal_sing"] == 1.0


def test_canonical_sing2():
    assert bench_canonical_sing2()["synthetic_canonical_sing2"] == 1.0


def test_klt_mmp():
    assert bench_klt_mmp()["synthetic_klt_mmp"] == 1.0


def test_mmp_flip():
    assert bench_mmp_flip()["synthetic_mmp_flip"] == 1.0


def test_abundance_conj():
    assert bench_abundance_conj()["synthetic_abundance_conj"] == 1.0


def test_bdd_fano():
    assert bench_bdd_fano()["synthetic_bdd_fano"] == 1.0
