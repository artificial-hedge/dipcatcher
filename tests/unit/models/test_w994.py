"""Wave-994 inverse-spectral canon tests."""

from __future__ import annotations

from quant_fund.models.borg_levinson import bench_borg_levinson
from quant_fund.models.gelfand_levitan import bench_gelfand_levitan
from quant_fund.models.inverse_scattering import bench_inverse_scattering
from quant_fund.models.kdv_isospectral import bench_kdv_isospectral
from quant_fund.models.marchenko_eq import bench_marchenko_eq
from quant_fund.models.trace_formulas import bench_trace_formulas


def test_inverse_scattering():
    assert bench_inverse_scattering()["synthetic_inverse_scattering"] == 1.0


def test_marchenko_eq():
    assert bench_marchenko_eq()["synthetic_marchenko_eq"] == 1.0


def test_gelfand_levitan():
    assert bench_gelfand_levitan()["synthetic_gelfand_levitan"] == 1.0


def test_kdv_isospectral():
    assert bench_kdv_isospectral()["synthetic_kdv_isospectral"] == 1.0


def test_trace_formulas():
    assert bench_trace_formulas()["synthetic_trace_formulas"] == 1.0


def test_borg_levinson():
    assert bench_borg_levinson()["synthetic_borg_levinson"] == 1.0
