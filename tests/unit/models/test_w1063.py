"""Wave-1063 communications/media canon tests."""

from __future__ import annotations

from quant_fund.models.communication_theory import bench_communication_theory
from quant_fund.models.digital_media import bench_digital_media
from quant_fund.models.journalism import bench_journalism
from quant_fund.models.media_studies import bench_media_studies
from quant_fund.models.public_relations import bench_public_relations
from quant_fund.models.rhetoric import bench_rhetoric


def test_media_studies():
    assert bench_media_studies()["synthetic_media_studies"] == 1.0


def test_journalism():
    assert bench_journalism()["synthetic_journalism"] == 1.0


def test_public_relations():
    assert bench_public_relations()["synthetic_public_relations"] == 1.0


def test_rhetoric():
    assert bench_rhetoric()["synthetic_rhetoric"] == 1.0


def test_communication_theory():
    assert bench_communication_theory()["synthetic_communication_theory"] == 1.0


def test_digital_media():
    assert bench_digital_media()["synthetic_digital_media"] == 1.0
