"""Wave-1131 sociology-5 canon tests."""

from __future__ import annotations

from quant_fund.models.digital_sociology import bench_digital_sociology
from quant_fund.models.sociology_of_disaster import bench_sociology_of_disaster
from quant_fund.models.sociology_of_housing import bench_sociology_of_housing
from quant_fund.models.sociology_of_migration import bench_sociology_of_migration
from quant_fund.models.sociology_of_risk import bench_sociology_of_risk
from quant_fund.models.sociology_of_the_body import bench_sociology_of_the_body


def test_sociology_of_migration():
    assert bench_sociology_of_migration()["synthetic_sociology_of_migration"] == 1.0


def test_sociology_of_housing():
    assert bench_sociology_of_housing()["synthetic_sociology_of_housing"] == 1.0


def test_sociology_of_disaster():
    assert bench_sociology_of_disaster()["synthetic_sociology_of_disaster"] == 1.0


def test_sociology_of_the_body():
    assert bench_sociology_of_the_body()["synthetic_sociology_of_the_body"] == 1.0


def test_sociology_of_risk():
    assert bench_sociology_of_risk()["synthetic_sociology_of_risk"] == 1.0


def test_digital_sociology():
    assert bench_digital_sociology()["synthetic_digital_sociology"] == 1.0
