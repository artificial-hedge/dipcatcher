"""Wave-1222 ortho canon tests."""

from __future__ import annotations

from quant_fund.models.hand_surgery import bench_hand_surgery
from quant_fund.models.joint_replacement import bench_joint_replacement
from quant_fund.models.musculoskeletal_medicine import bench_musculoskeletal_medicine
from quant_fund.models.orthopedics_studies import bench_orthopedics_studies
from quant_fund.models.spine_surgery import bench_spine_surgery
from quant_fund.models.sports_medicine_orthopedics import bench_sports_medicine_orthopedics


def test_orthopedics_studies():
    assert bench_orthopedics_studies()["synthetic_orthopedics_studies"] == 1.0


def test_sports_medicine_orthopedics():
    assert bench_sports_medicine_orthopedics()["synthetic_sports_medicine_orthopedics"] == 1.0


def test_musculoskeletal_medicine():
    assert bench_musculoskeletal_medicine()["synthetic_musculoskeletal_medicine"] == 1.0


def test_spine_surgery():
    assert bench_spine_surgery()["synthetic_spine_surgery"] == 1.0


def test_joint_replacement():
    assert bench_joint_replacement()["synthetic_joint_replacement"] == 1.0


def test_hand_surgery():
    assert bench_hand_surgery()["synthetic_hand_surgery"] == 1.0
