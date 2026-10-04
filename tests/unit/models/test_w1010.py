"""Wave-1010 quantum-field-theory canon tests."""

from __future__ import annotations

from quant_fund.models.canonical_quantization import bench_canonical_quantization
from quant_fund.models.dirac_equation import bench_dirac_equation
from quant_fund.models.feynman_rules import bench_feynman_rules
from quant_fund.models.klein_gordon import bench_klein_gordon
from quant_fund.models.path_integral_qm import bench_path_integral_qm
from quant_fund.models.renormalization_group import bench_renormalization_group


def test_klein_gordon():
    assert bench_klein_gordon()["synthetic_klein_gordon"] == 1.0


def test_dirac_equation():
    assert bench_dirac_equation()["synthetic_dirac_equation"] == 1.0


def test_feynman_rules():
    assert bench_feynman_rules()["synthetic_feynman_rules"] == 1.0


def test_renormalization_group():
    assert bench_renormalization_group()["synthetic_renormalization_group"] == 1.0


def test_path_integral_qm():
    assert bench_path_integral_qm()["synthetic_path_integral_qm"] == 1.0


def test_canonical_quantization():
    assert bench_canonical_quantization()["synthetic_canonical_quantization"] == 1.0
