import pytest


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_gaming_studies",
        "benchmark_saturate_studies",
        "contamination_studies",
        "eval_coverage_studies",
        "eval_reliability_studies",
        "lm_eval_harness_studies",
    ],
)
def test_w1297_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
