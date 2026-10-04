import pytest


@pytest.mark.parametrize(
    "name",
    [
        "boolq_lite_studies",
        "race_lite_studies",
        "cosmos_qa_studies",
        "social_qa_studies",
        "arc_easy2_studies",
        "sciq_lite_studies",
    ],
)
def test_w1354_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
