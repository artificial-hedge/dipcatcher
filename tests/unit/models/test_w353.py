from quant_fund.models.gronwall_lemma import bench_gronwall_lemma
from quant_fund.models.lyapunov_stability import bench_lyapunov_stability
from quant_fund.models.phase_plane import bench_phase_plane
from quant_fund.models.picard_lindelof import bench_picard_lindelof
from quant_fund.models.sturm_liouville import bench_sturm_liouville
from quant_fund.models.variation_params import bench_variation_params


def test_picard_lindelof():
    assert bench_picard_lindelof()["synthetic_picard_lindelof"] == 1.0


def test_gronwall_lemma():
    assert bench_gronwall_lemma()["synthetic_gronwall_lemma"] == 1.0


def test_sturm_liouville():
    assert bench_sturm_liouville()["synthetic_sturm_liouville"] == 1.0


def test_phase_plane():
    assert bench_phase_plane()["synthetic_phase_plane"] == 1.0


def test_lyapunov_stability():
    assert bench_lyapunov_stability()["synthetic_lyapunov_stability"] == 1.0


def test_variation_params():
    assert bench_variation_params()["synthetic_variation_params"] == 1.0
