from quant_fund.models.furstenberg import bench_furstenberg
from quant_fund.models.gallai_thm import bench_gallai_thm
from quant_fund.models.hales_jewett import bench_hales_jewett
from quant_fund.models.hindman import bench_hindman
from quant_fund.models.rado_thm import bench_rado_thm
from quant_fund.models.schur_thm import bench_schur_thm


def test_hales_jewett():
    assert bench_hales_jewett()["synthetic_hales_jewett"] == 1.0


def test_rado_thm():
    assert bench_rado_thm()["synthetic_rado_thm"] == 1.0


def test_gallai_thm():
    assert bench_gallai_thm()["synthetic_gallai_thm"] == 1.0


def test_schur_thm():
    assert bench_schur_thm()["synthetic_schur_thm"] == 1.0


def test_hindman():
    assert bench_hindman()["synthetic_hindman"] == 1.0


def test_furstenberg():
    assert bench_furstenberg()["synthetic_furstenberg"] == 1.0
