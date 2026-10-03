"""Wave-922 distributed-systems-6 canon tests."""

from __future__ import annotations

from quant_fund.models.abcast_lite import bench_abcast_lite
from quant_fund.models.cap_theorem import bench_cap_theorem
from quant_fund.models.cbc_bcast import bench_cbc_bcast
from quant_fund.models.lake_wisc import bench_lake_wisc
from quant_fund.models.slush_consensus import bench_slush_consensus
from quant_fund.models.snowflake_consensus import bench_snowflake_consensus


def test_abcast_lite():
    assert bench_abcast_lite()["synthetic_abcast_lite"] == 1.0


def test_cbc_bcast():
    assert bench_cbc_bcast()["synthetic_cbc_bcast"] == 1.0


def test_slush_consensus():
    assert bench_slush_consensus()["synthetic_slush_consensus"] == 1.0


def test_snowflake_consensus():
    assert bench_snowflake_consensus()["synthetic_snowflake_consensus"] == 1.0


def test_cap_theorem():
    assert bench_cap_theorem()["synthetic_cap_theorem"] == 1.0


def test_lake_wisc():
    assert bench_lake_wisc()["synthetic_lake_wisc"] == 1.0
