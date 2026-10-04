from quant_fund.models.cofibrant_rep import bench_cofibrant_rep
from quant_fund.models.enriched_model import bench_enriched_model
from quant_fund.models.localization_mc import (
    bench_localization_mc,
)
from quant_fund.models.monoidal_model import bench_monoidal_model
from quant_fund.models.quillen_equiv import bench_quillen_equiv
from quant_fund.models.reedy_model import bench_reedy_model


def test_cofibrant_rep():
    assert bench_cofibrant_rep()["synthetic_cofibrant_rep"] == 1.0


def test_quillen_equiv():
    assert bench_quillen_equiv()["synthetic_quillen_equiv"] == 1.0


def test_monoidal_model():
    assert bench_monoidal_model()["synthetic_monoidal_model"] == 1.0


def test_enriched_model():
    assert bench_enriched_model()["synthetic_enriched_model"] == 1.0


def test_reedy_model():
    assert bench_reedy_model()["synthetic_reedy_model"] == 1.0


def test_localization_mc():
    assert bench_localization_mc()["synthetic_localization_mc"] == 1.0
