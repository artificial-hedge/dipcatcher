"""Wave-1202 infectious-disease canon tests."""

from __future__ import annotations

from quant_fund.models.entomology_medical import bench_entomology_medical
from quant_fund.models.medical_microbiology import bench_medical_microbiology
from quant_fund.models.mycology_studies import bench_mycology_studies
from quant_fund.models.parasitology_studies import bench_parasitology_studies
from quant_fund.models.public_health_microbiology import bench_public_health_microbiology
from quant_fund.models.vector_borne_diseases import bench_vector_borne_diseases


def test_public_health_microbiology():
    assert bench_public_health_microbiology()["synthetic_public_health_microbiology"] == 1.0


def test_medical_microbiology():
    assert bench_medical_microbiology()["synthetic_medical_microbiology"] == 1.0


def test_parasitology_studies():
    assert bench_parasitology_studies()["synthetic_parasitology_studies"] == 1.0


def test_mycology_studies():
    assert bench_mycology_studies()["synthetic_mycology_studies"] == 1.0


def test_entomology_medical():
    assert bench_entomology_medical()["synthetic_entomology_medical"] == 1.0


def test_vector_borne_diseases():
    assert bench_vector_borne_diseases()["synthetic_vector_borne_diseases"] == 1.0
