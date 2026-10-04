import pytest


@pytest.mark.parametrize(
    "name",
    [
        "touch_e_studies",
        "fever_lite_studies",
        "climate_fever_studies",
        "verdict_qa_studies",
        "cite_worth_studies",
        "bioasq_lite_studies",
    ],
)
def test_w1360_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
