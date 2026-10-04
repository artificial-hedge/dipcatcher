import pytest


@pytest.mark.parametrize(
    "name",
    [
        "crake_qa_studies",
        "dabchick_qa_studies",
        "gallinule_qa_studies",
        "rail_qa_studies",
        "waterhen_qa_studies",
        "coot_qa_studies",
    ],
)
def test_w1537_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
