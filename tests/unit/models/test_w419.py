from quant_fund.models.bundle_section import bench_bundle_section
from quant_fund.models.classify_space import bench_classify_space
from quant_fund.models.path_fibration import bench_path_fibration
from quant_fund.models.serre_fibration import bench_serre_fibration
from quant_fund.models.thom_space import bench_thom_space
from quant_fund.models.vector_bundle import bench_vector_bundle


def test_serre_fibration():
    assert bench_serre_fibration()["synthetic_serre_fibration"] == 1.0


def test_path_fibration():
    assert bench_path_fibration()["synthetic_path_fibration"] == 1.0


def test_bundle_section():
    assert bench_bundle_section()["synthetic_bundle_section"] == 1.0


def test_classify_space():
    assert bench_classify_space()["synthetic_classify_space"] == 1.0


def test_vector_bundle():
    assert bench_vector_bundle()["synthetic_vector_bundle"] == 1.0


def test_thom_space():
    assert bench_thom_space()["synthetic_thom_space"] == 1.0
