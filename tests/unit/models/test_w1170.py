"""Wave-1170 justice canon tests."""

from __future__ import annotations

from quant_fund.models.criminology_3 import bench_criminology_3
from quant_fund.models.forensic_science_2 import bench_forensic_science_2
from quant_fund.models.intelligence_studies_2 import bench_intelligence_studies_2
from quant_fund.models.penology_2 import bench_penology_2
from quant_fund.models.security_studies_2 import bench_security_studies_2
from quant_fund.models.victimology_2 import bench_victimology_2


def test_criminology_3():
    assert bench_criminology_3()["synthetic_criminology_3"] == 1.0


def test_forensic_science_2():
    assert bench_forensic_science_2()["synthetic_forensic_science_2"] == 1.0


def test_penology_2():
    assert bench_penology_2()["synthetic_penology_2"] == 1.0


def test_victimology_2():
    assert bench_victimology_2()["synthetic_victimology_2"] == 1.0


def test_security_studies_2():
    assert bench_security_studies_2()["synthetic_security_studies_2"] == 1.0


def test_intelligence_studies_2():
    assert bench_intelligence_studies_2()["synthetic_intelligence_studies_2"] == 1.0
