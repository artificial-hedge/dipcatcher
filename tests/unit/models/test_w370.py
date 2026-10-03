from quant_fund.models.dirac_ore import bench_dirac_ore
from quant_fund.models.graph_minor import bench_graph_minor
from quant_fund.models.planar_five import bench_planar_five
from quant_fund.models.ramsey_num import bench_ramsey_num
from quant_fund.models.turan_theorem import bench_turan_theorem
from quant_fund.models.tutte_berge import bench_tutte_berge


def test_tutte_berge():
    assert bench_tutte_berge()["synthetic_tutte_berge"] == 1.0


def test_dirac_ore():
    assert bench_dirac_ore()["synthetic_dirac_ore"] == 1.0


def test_turan_theorem():
    assert bench_turan_theorem()["synthetic_turan_theorem"] == 1.0


def test_planar_five():
    assert bench_planar_five()["synthetic_planar_five"] == 1.0


def test_graph_minor():
    assert bench_graph_minor()["synthetic_graph_minor"] == 1.0


def test_ramsey_num():
    assert bench_ramsey_num()["synthetic_ramsey_num"] == 1.0
