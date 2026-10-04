import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mmmu_studies",
        "mathvista_studies",
        "videomme_studies",
        "chart_gqa_studies",
        "mmmlu_studies",
        "mkqa_studies",
    ],
)
def test_w1321_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
