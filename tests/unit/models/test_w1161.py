"""Wave-1161 health-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.dentistry_3 import bench_dentistry_3
from quant_fund.models.medicine_7 import bench_medicine_7
from quant_fund.models.nursing_2 import bench_nursing_2
from quant_fund.models.pharmacy_2 import bench_pharmacy_2
from quant_fund.models.public_health_2 import bench_public_health_2
from quant_fund.models.veterinary_medicine_2 import bench_veterinary_medicine_2


def test_medicine_7():
    assert bench_medicine_7()["synthetic_medicine_7"] == 1.0


def test_dentistry_3():
    assert bench_dentistry_3()["synthetic_dentistry_3"] == 1.0


def test_nursing_2():
    assert bench_nursing_2()["synthetic_nursing_2"] == 1.0


def test_public_health_2():
    assert bench_public_health_2()["synthetic_public_health_2"] == 1.0


def test_veterinary_medicine_2():
    assert bench_veterinary_medicine_2()["synthetic_veterinary_medicine_2"] == 1.0


def test_pharmacy_2():
    assert bench_pharmacy_2()["synthetic_pharmacy_2"] == 1.0
