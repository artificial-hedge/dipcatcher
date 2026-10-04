"""Wave-1073 theology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.biblical_exegesis import bench_biblical_exegesis
from quant_fund.models.church_history import bench_church_history
from quant_fund.models.liturgical_studies import bench_liturgical_studies
from quant_fund.models.missiology import bench_missiology
from quant_fund.models.pastoral_theology import bench_pastoral_theology
from quant_fund.models.systematic_theology import bench_systematic_theology


def test_systematic_theology():
    assert bench_systematic_theology()["synthetic_systematic_theology"] == 1.0


def test_biblical_exegesis():
    assert bench_biblical_exegesis()["synthetic_biblical_exegesis"] == 1.0


def test_church_history():
    assert bench_church_history()["synthetic_church_history"] == 1.0


def test_pastoral_theology():
    assert bench_pastoral_theology()["synthetic_pastoral_theology"] == 1.0


def test_liturgical_studies():
    assert bench_liturgical_studies()["synthetic_liturgical_studies"] == 1.0


def test_missiology():
    assert bench_missiology()["synthetic_missiology"] == 1.0
