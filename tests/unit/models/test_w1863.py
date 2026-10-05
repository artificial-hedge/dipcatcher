import pytest


@pytest.mark.parametrize(
    "name",
    [
        "uther_qa_studies",
        "lot_qa_studies",
        "marhaus_qa_studies",
        "balin_qa_studies",
        "pellinor_qa_studies",
        "lamorak_qa_studies",
    ],
)
def test_w1863_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
