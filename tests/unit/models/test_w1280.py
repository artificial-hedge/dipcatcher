import pytest


@pytest.mark.parametrize(
    "name",
    [
        "data_mixture_studies",
        "data_quality_studies",
        "dedup_pipeline_studies",
        "domain_filtering_studies",
        "synthetic_data_studies",
        "token_budget_studies",
    ],
)
def test_w1280_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
