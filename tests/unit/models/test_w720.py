from quant_fund.models.categorical_entropy import (
    bench_categorical_entropy,
)
from quant_fund.models.cluster_tilting import (
    bench_cluster_tilting,
)
from quant_fund.models.derived_morita import (
    bench_derived_morita,
)
from quant_fund.models.preprojective_alg import (
    bench_preprojective_alg,
)
from quant_fund.models.rouquier_dim import bench_rouquier_dim
from quant_fund.models.serre_dim import bench_serre_dim


def test_cluster_tilting():
    assert bench_cluster_tilting()["synthetic_cluster_tilting"] == 1.0


def test_derived_morita():
    assert bench_derived_morita()["synthetic_derived_morita"] == 1.0


def test_preprojective_alg():
    assert bench_preprojective_alg()["synthetic_preprojective_alg"] == 1.0


def test_categorical_entropy():
    assert bench_categorical_entropy()["synthetic_categorical_entropy"] == 1.0


def test_serre_dim():
    assert bench_serre_dim()["synthetic_serre_dim"] == 1.0


def test_rouquier_dim():
    assert bench_rouquier_dim()["synthetic_rouquier_dim"] == 1.0
