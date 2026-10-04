from quant_fund.models.diffuse_element import (
    bench_diffuse_element,
)
from quant_fund.models.hp_clouds import (
    bench_hp_clouds,
)
from quant_fund.models.meshless_local import (
    bench_meshless_local,
)
from quant_fund.models.mls_shape import (
    bench_mls_shape,
)
from quant_fund.models.moving_least_sq import (
    bench_moving_least_sq,
)
from quant_fund.models.point_cloud_interp import (
    bench_point_cloud_interp,
)


def test_moving_least_sq():
    assert bench_moving_least_sq()["synthetic_moving_least_sq"] == 1.0


def test_mls_shape():
    assert bench_mls_shape()["synthetic_mls_shape"] == 1.0


def test_hp_clouds():
    assert bench_hp_clouds()["synthetic_hp_clouds"] == 1.0


def test_meshless_local():
    assert bench_meshless_local()["synthetic_meshless_local"] == 1.0


def test_point_cloud_interp():
    assert bench_point_cloud_interp()["synthetic_point_cloud_interp"] == 1.0


def test_diffuse_element():
    assert bench_diffuse_element()["synthetic_diffuse_element"] == 1.0
