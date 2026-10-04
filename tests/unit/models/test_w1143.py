"""Wave-1143 physics-5 canon tests."""

from __future__ import annotations

from quant_fund.models.acoustics_2 import bench_acoustics_2
from quant_fund.models.biophysics_2 import bench_biophysics_2
from quant_fund.models.condensed_matter_3 import bench_condensed_matter_3
from quant_fund.models.nanotechnology import bench_nanotechnology
from quant_fund.models.optics_3 import bench_optics_3
from quant_fund.models.thermodynamics_2 import bench_thermodynamics_2


def test_nanotechnology():
    assert bench_nanotechnology()["synthetic_nanotechnology"] == 1.0


def test_biophysics_2():
    assert bench_biophysics_2()["synthetic_biophysics_2"] == 1.0


def test_condensed_matter_3():
    assert bench_condensed_matter_3()["synthetic_condensed_matter_3"] == 1.0


def test_optics_3():
    assert bench_optics_3()["synthetic_optics_3"] == 1.0


def test_acoustics_2():
    assert bench_acoustics_2()["synthetic_acoustics_2"] == 1.0


def test_thermodynamics_2():
    assert bench_thermodynamics_2()["synthetic_thermodynamics_2"] == 1.0
