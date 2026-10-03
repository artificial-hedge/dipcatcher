from quant_fund.models.chiral_alg import bench_chiral_alg
from quant_fund.models.cyclic_hk import bench_cyclic_hk
from quant_fund.models.dendroidal import bench_dendroidal
from quant_fund.models.infty_operad import bench_infty_operad
from quant_fund.models.seq_spectra import bench_seq_spectra
from quant_fund.models.sifted_cat import bench_sifted_cat


def test_dendroidal():
    assert bench_dendroidal()["synthetic_dendroidal"] == 1.0


def test_infty_operad():
    assert bench_infty_operad()["synthetic_infty_operad"] == 1.0


def test_cyclic_hk():
    assert bench_cyclic_hk()["synthetic_cyclic_hk"] == 1.0


def test_chiral_alg():
    assert bench_chiral_alg()["synthetic_chiral_alg"] == 1.0


def test_sifted_cat():
    assert bench_sifted_cat()["synthetic_sifted_cat"] == 1.0


def test_seq_spectra():
    assert bench_seq_spectra()["synthetic_seq_spectra"] == 1.0
