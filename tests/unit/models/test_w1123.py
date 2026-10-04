"""Wave-1123 linguistics-5 canon tests."""

from __future__ import annotations

from quant_fund.models.contact_linguistics import bench_contact_linguistics
from quant_fund.models.descriptive_linguistics import bench_descriptive_linguistics
from quant_fund.models.dialectometry import bench_dialectometry
from quant_fund.models.etymology import bench_etymology
from quant_fund.models.lexicography import bench_lexicography
from quant_fund.models.philological_studies import bench_philological_studies


def test_contact_linguistics():
    assert bench_contact_linguistics()["synthetic_contact_linguistics"] == 1.0


def test_descriptive_linguistics():
    assert bench_descriptive_linguistics()["synthetic_descriptive_linguistics"] == 1.0


def test_philological_studies():
    assert bench_philological_studies()["synthetic_philological_studies"] == 1.0


def test_etymology():
    assert bench_etymology()["synthetic_etymology"] == 1.0


def test_dialectometry():
    assert bench_dialectometry()["synthetic_dialectometry"] == 1.0


def test_lexicography():
    assert bench_lexicography()["synthetic_lexicography"] == 1.0
