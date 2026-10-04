"""Wave-1069 criminal justice canon tests."""

from __future__ import annotations

from quant_fund.models.criminal_justice import bench_criminal_justice
from quant_fund.models.criminal_procedure import bench_criminal_procedure
from quant_fund.models.forensic_science import bench_forensic_science
from quant_fund.models.penology import bench_penology
from quant_fund.models.policing_studies import bench_policing_studies
from quant_fund.models.victimology import bench_victimology


def test_criminal_justice():
    assert bench_criminal_justice()["synthetic_criminal_justice"] == 1.0


def test_forensic_science():
    assert bench_forensic_science()["synthetic_forensic_science"] == 1.0


def test_penology():
    assert bench_penology()["synthetic_penology"] == 1.0


def test_policing_studies():
    assert bench_policing_studies()["synthetic_policing_studies"] == 1.0


def test_victimology():
    assert bench_victimology()["synthetic_victimology"] == 1.0


def test_criminal_procedure():
    assert bench_criminal_procedure()["synthetic_criminal_procedure"] == 1.0
