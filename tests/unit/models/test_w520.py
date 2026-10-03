from quant_fund.models.converse_thm import bench_converse_thm
from quant_fund.models.gln_automorphic import bench_gln_automorphic
from quant_fund.models.godement_jacq import bench_godement_jacq
from quant_fund.models.langlands_lfunc import bench_langlands_lfunc
from quant_fund.models.rankin_selberg import bench_rankin_selberg
from quant_fund.models.whittaker_model import bench_whittaker_model


def test_gln_automorphic():
    assert bench_gln_automorphic()["synthetic_gln_automorphic"] == 1.0


def test_whittaker_model():
    assert bench_whittaker_model()["synthetic_whittaker_model"] == 1.0


def test_godement_jacq():
    assert bench_godement_jacq()["synthetic_godement_jacq"] == 1.0


def test_rankin_selberg():
    assert bench_rankin_selberg()["synthetic_rankin_selberg"] == 1.0


def test_langlands_lfunc():
    assert bench_langlands_lfunc()["synthetic_langlands_lfunc"] == 1.0


def test_converse_thm():
    assert bench_converse_thm()["synthetic_converse_thm"] == 1.0
