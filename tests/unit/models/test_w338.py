from quant_fund.models.ac_choice import bench_ac_choice
from quant_fund.models.cardinal_arith import bench_cardinal_arith
from quant_fund.models.ordinal_arith import bench_ordinal_arith
from quant_fund.models.transfinite_induct import bench_transfinite_induct
from quant_fund.models.v_omega import bench_v_omega
from quant_fund.models.well_founded import bench_well_founded


def test_ordinal_arith():
    assert bench_ordinal_arith()["synthetic_ordinal_arith"] == 1.0


def test_cardinal_arith():
    assert bench_cardinal_arith()["synthetic_cardinal_arith"] == 1.0


def test_transfinite_induct():
    assert bench_transfinite_induct()["synthetic_transfinite_induct"] == 1.0


def test_well_founded():
    assert bench_well_founded()["synthetic_well_founded"] == 1.0


def test_v_omega():
    assert bench_v_omega()["synthetic_v_omega"] == 1.0


def test_ac_choice():
    assert bench_ac_choice()["synthetic_ac_choice"] == 1.0
