from quant_fund.models.cw_complex import bench_cw_complex
from quant_fund.models.excision import bench_excision
from quant_fund.models.homotopy_group import bench_homotopy_group
from quant_fund.models.poincare_dual import bench_poincare_dual
from quant_fund.models.singular_homology import bench_singular_homology
from quant_fund.models.spectral_seq_toy import bench_spectral_seq_toy


def test_singular_homology():
    assert bench_singular_homology()["synthetic_singular_homology"] == 1.0


def test_cw_complex():
    assert bench_cw_complex()["synthetic_cw_complex"] == 1.0


def test_spectral_seq_toy():
    assert bench_spectral_seq_toy()["synthetic_spectral_seq_toy"] == 1.0


def test_homotopy_group():
    assert bench_homotopy_group()["synthetic_homotopy_group"] == 1.0


def test_excision():
    assert bench_excision()["synthetic_excision"] == 1.0


def test_poincare_dual():
    assert bench_poincare_dual()["synthetic_poincare_dual"] == 1.0
