"""Wave-1104 political-science-2 canon tests."""

from __future__ import annotations

from quant_fund.models.american_politics import bench_american_politics
from quant_fund.models.policy_analysis import bench_policy_analysis
from quant_fund.models.political_behavior import bench_political_behavior
from quant_fund.models.political_methodology import bench_political_methodology
from quant_fund.models.public_law import bench_public_law
from quant_fund.models.security_studies import bench_security_studies


def test_american_politics():
    assert bench_american_politics()["synthetic_american_politics"] == 1.0


def test_political_behavior():
    assert bench_political_behavior()["synthetic_political_behavior"] == 1.0


def test_public_law():
    assert bench_public_law()["synthetic_public_law"] == 1.0


def test_political_methodology():
    assert bench_political_methodology()["synthetic_political_methodology"] == 1.0


def test_security_studies():
    assert bench_security_studies()["synthetic_security_studies"] == 1.0


def test_policy_analysis():
    assert bench_policy_analysis()["synthetic_policy_analysis"] == 1.0
