"""Wave-976 nuclear-spaces canon tests."""

from __future__ import annotations

from quant_fund.models.diam_dim import bench_diam_dim
from quant_fund.models.frechet_nuclear import bench_frechet_nuclear
from quant_fund.models.gelfand_triple import bench_gelfand_triple
from quant_fund.models.hilbert_schmidt_emb import bench_hilbert_schmidt_emb
from quant_fund.models.nuclear_map import bench_nuclear_map
from quant_fund.models.trace_duality import bench_trace_duality


def test_nuclear_map():
    assert bench_nuclear_map()["synthetic_nuclear_map"] == 1.0


def test_frechet_nuclear():
    assert bench_frechet_nuclear()["synthetic_frechet_nuclear"] == 1.0


def test_gelfand_triple():
    assert bench_gelfand_triple()["synthetic_gelfand_triple"] == 1.0


def test_hilbert_schmidt_emb():
    assert bench_hilbert_schmidt_emb()["synthetic_hilbert_schmidt_emb"] == 1.0


def test_trace_duality():
    assert bench_trace_duality()["synthetic_trace_duality"] == 1.0


def test_diam_dim():
    assert bench_diam_dim()["synthetic_diam_dim"] == 1.0
