from quant_fund.models.boneschi_boal import bench_boneschi_boal
from quant_fund.models.bradley_mixing import bench_bradley_mixing
from quant_fund.models.hopf_chain import bench_hopf_chain
from quant_fund.models.ibagimov_mixing import bench_ibagimov_mixing
from quant_fund.models.polya_urn import bench_polya_urn
from quant_fund.models.rosenthal_mom import bench_rosenthal_mom


def test_polya_urn():
    assert bench_polya_urn()["synthetic_polya_urn"] == 1.0


def test_hopf_chain():
    assert bench_hopf_chain()["synthetic_hopf_chain"] == 1.0


def test_boneschi_boal():
    assert bench_boneschi_boal()["synthetic_boneschi_boal"] == 1.0


def test_bradley_mixing():
    assert bench_bradley_mixing()["synthetic_bradley_mixing"] == 1.0


def test_rosenthal_mom():
    assert bench_rosenthal_mom()["synthetic_rosenthal_mom"] == 1.0


def test_ibagimov_mixing():
    assert bench_ibagimov_mixing()["synthetic_ibagimov_mixing"] == 1.0
