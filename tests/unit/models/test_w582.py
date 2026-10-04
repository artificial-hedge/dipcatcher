from quant_fund.models.arinkin_gaitsgory import (
    bench_arinkin_gaitsgory,
)
from quant_fund.models.derived_satake import (
    bench_derived_satake,
)
from quant_fund.models.fusion_product import (
    bench_fusion_product,
)
from quant_fund.models.geometric_satake2 import (
    bench_geometric_satake2,
)
from quant_fund.models.nilp_cone import bench_nilp_cone
from quant_fund.models.spectral_bung import bench_spectral_bung


def test_arinkin_gaitsgory():
    assert bench_arinkin_gaitsgory()["synthetic_arinkin_gaitsgory"] == 1.0


def test_derived_satake():
    assert bench_derived_satake()["synthetic_derived_satake"] == 1.0


def test_spectral_bung():
    assert bench_spectral_bung()["synthetic_spectral_bung"] == 1.0


def test_nilp_cone():
    assert bench_nilp_cone()["synthetic_nilp_cone"] == 1.0


def test_geometric_satake2():
    assert bench_geometric_satake2()["synthetic_geometric_satake2"] == 1.0


def test_fusion_product():
    assert bench_fusion_product()["synthetic_fusion_product"] == 1.0
