"""Wave-1177 formal-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.axiomatic_systems import bench_axiomatic_systems
from quant_fund.models.formal_ontology import bench_formal_ontology
from quant_fund.models.formal_sciences import bench_formal_sciences
from quant_fund.models.mathematical_logic import bench_mathematical_logic
from quant_fund.models.model_checking_2 import bench_model_checking_2
from quant_fund.models.proof_calculus import bench_proof_calculus


def test_formal_sciences():
    assert bench_formal_sciences()["synthetic_formal_sciences"] == 1.0


def test_mathematical_logic():
    assert bench_mathematical_logic()["synthetic_mathematical_logic"] == 1.0


def test_axiomatic_systems():
    assert bench_axiomatic_systems()["synthetic_axiomatic_systems"] == 1.0


def test_proof_calculus():
    assert bench_proof_calculus()["synthetic_proof_calculus"] == 1.0


def test_model_checking_2():
    assert bench_model_checking_2()["synthetic_model_checking_2"] == 1.0


def test_formal_ontology():
    assert bench_formal_ontology()["synthetic_formal_ontology"] == 1.0
