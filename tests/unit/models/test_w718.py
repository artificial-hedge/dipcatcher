from quant_fund.models.der_bimodule import bench_der_bimodule
from quant_fund.models.helix_theory import bench_helix_theory
from quant_fund.models.higher_auslander import (
    bench_higher_auslander,
)
from quant_fund.models.icy_paper import bench_icy_paper
from quant_fund.models.mutation_class import (
    bench_mutation_class,
)
from quant_fund.models.rep_finite import bench_rep_finite


def test_helix_theory():
    assert bench_helix_theory()["synthetic_helix_theory"] == 1.0


def test_mutation_class():
    assert bench_mutation_class()["synthetic_mutation_class"] == 1.0


def test_rep_finite():
    assert bench_rep_finite()["synthetic_rep_finite"] == 1.0


def test_der_bimodule():
    assert bench_der_bimodule()["synthetic_der_bimodule"] == 1.0


def test_icy_paper():
    assert bench_icy_paper()["synthetic_icy_paper"] == 1.0


def test_higher_auslander():
    assert bench_higher_auslander()["synthetic_higher_auslander"] == 1.0
