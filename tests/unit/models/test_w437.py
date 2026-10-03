from quant_fund.models.comparison_iso import bench_comparison_iso
from quant_fund.models.crystalline_coh import bench_crystalline_coh
from quant_fund.models.derham_coh import bench_derham_coh
from quant_fund.models.etale_coh import bench_etale_coh
from quant_fund.models.frobenius_coh import bench_frobenius_coh
from quant_fund.models.prismatic_coh import bench_prismatic_coh


def test_crystalline_coh():
    assert bench_crystalline_coh()["synthetic_crystalline_coh"] == 1.0


def test_prismatic_coh():
    assert bench_prismatic_coh()["synthetic_prismatic_coh"] == 1.0


def test_etale_coh():
    assert bench_etale_coh()["synthetic_etale_coh"] == 1.0


def test_derham_coh():
    assert bench_derham_coh()["synthetic_derham_coh"] == 1.0


def test_frobenius_coh():
    assert bench_frobenius_coh()["synthetic_frobenius_coh"] == 1.0


def test_comparison_iso():
    assert bench_comparison_iso()["synthetic_comparison_iso"] == 1.0
