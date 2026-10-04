"""Wave-1186 ux canon tests."""

from __future__ import annotations

from quant_fund.models.accessibility_studies import bench_accessibility_studies
from quant_fund.models.hci_studies import bench_hci_studies
from quant_fund.models.information_architecture import bench_information_architecture
from quant_fund.models.interaction_design import bench_interaction_design
from quant_fund.models.service_design import bench_service_design
from quant_fund.models.ux_design import bench_ux_design


def test_ux_design():
    assert bench_ux_design()["synthetic_ux_design"] == 1.0


def test_hci_studies():
    assert bench_hci_studies()["synthetic_hci_studies"] == 1.0


def test_information_architecture():
    assert bench_information_architecture()["synthetic_information_architecture"] == 1.0


def test_interaction_design():
    assert bench_interaction_design()["synthetic_interaction_design"] == 1.0


def test_accessibility_studies():
    assert bench_accessibility_studies()["synthetic_accessibility_studies"] == 1.0


def test_service_design():
    assert bench_service_design()["synthetic_service_design"] == 1.0
