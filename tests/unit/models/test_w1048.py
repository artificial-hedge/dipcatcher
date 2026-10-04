"""Wave-1048 veterinary-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.animal_surgery import bench_animal_surgery
from quant_fund.models.equine_medicine import bench_equine_medicine
from quant_fund.models.veterinary_anatomy import bench_veterinary_anatomy
from quant_fund.models.veterinary_epidemiology import bench_veterinary_epidemiology
from quant_fund.models.veterinary_pathology import bench_veterinary_pathology
from quant_fund.models.veterinary_pharmacology import bench_veterinary_pharmacology


def test_veterinary_anatomy():
    assert bench_veterinary_anatomy()["synthetic_veterinary_anatomy"] == 1.0


def test_veterinary_pathology():
    assert bench_veterinary_pathology()["synthetic_veterinary_pathology"] == 1.0


def test_veterinary_pharmacology():
    assert bench_veterinary_pharmacology()["synthetic_veterinary_pharmacology"] == 1.0


def test_animal_surgery():
    assert bench_animal_surgery()["synthetic_animal_surgery"] == 1.0


def test_veterinary_epidemiology():
    assert bench_veterinary_epidemiology()["synthetic_veterinary_epidemiology"] == 1.0


def test_equine_medicine():
    assert bench_equine_medicine()["synthetic_equine_medicine"] == 1.0
