"""Wave-1093 law-2 canon tests."""

from __future__ import annotations

from quant_fund.models.canon_law import bench_canon_law
from quant_fund.models.civil_law import bench_civil_law
from quant_fund.models.common_law import bench_common_law
from quant_fund.models.maritime_law import bench_maritime_law
from quant_fund.models.procedural_law import bench_procedural_law
from quant_fund.models.property_law import bench_property_law


def test_civil_law():
    assert bench_civil_law()["synthetic_civil_law"] == 1.0


def test_common_law():
    assert bench_common_law()["synthetic_common_law"] == 1.0


def test_canon_law():
    assert bench_canon_law()["synthetic_canon_law"] == 1.0


def test_maritime_law():
    assert bench_maritime_law()["synthetic_maritime_law"] == 1.0


def test_property_law():
    assert bench_property_law()["synthetic_property_law"] == 1.0


def test_procedural_law():
    assert bench_procedural_law()["synthetic_procedural_law"] == 1.0
