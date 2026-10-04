from quant_fund.models.cut_elim_seq import bench_cut_elim_seq
from quant_fund.models.finitary_induct import bench_finitary_induct
from quant_fund.models.herbrand_thm import bench_herbrand_thm
from quant_fund.models.hilbert_system import bench_hilbert_system
from quant_fund.models.interp_equality import bench_interp_equality
from quant_fund.models.reverse_math import bench_reverse_math


def test_herbrand_thm():
    assert bench_herbrand_thm()["synthetic_herbrand_thm"] == 1.0


def test_interp_equality():
    assert bench_interp_equality()["synthetic_interp_equality"] == 1.0


def test_cut_elim_seq():
    assert bench_cut_elim_seq()["synthetic_cut_elim_seq"] == 1.0


def test_finitary_induct():
    assert bench_finitary_induct()["synthetic_finitary_induct"] == 1.0


def test_hilbert_system():
    assert bench_hilbert_system()["synthetic_hilbert_system"] == 1.0


def test_reverse_math():
    assert bench_reverse_math()["synthetic_reverse_math"] == 1.0
