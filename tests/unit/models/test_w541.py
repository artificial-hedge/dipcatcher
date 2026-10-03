from quant_fund.models.jensen_formula import bench_jensen_formula
from quant_fund.models.montel_normal import bench_montel_normal
from quant_fund.models.picard_thm import bench_picard_thm
from quant_fund.models.riemann_mapping import bench_riemann_mapping
from quant_fund.models.runge_approx import bench_runge_approx
from quant_fund.models.schwarz_lemma import bench_schwarz_lemma


def test_riemann_mapping():
    assert bench_riemann_mapping()["synthetic_riemann_mapping"] == 1.0


def test_schwarz_lemma():
    assert bench_schwarz_lemma()["synthetic_schwarz_lemma"] == 1.0


def test_picard_thm():
    assert bench_picard_thm()["synthetic_picard_thm"] == 1.0


def test_montel_normal():
    assert bench_montel_normal()["synthetic_montel_normal"] == 1.0


def test_runge_approx():
    assert bench_runge_approx()["synthetic_runge_approx"] == 1.0


def test_jensen_formula():
    assert bench_jensen_formula()["synthetic_jensen_formula"] == 1.0
