import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bluff_qa_studies",
        "headland_qa_studies",
        "cove_qa_studies",
        "inlet_qa_studies",
        "islet_qa_studies",
        "atoll_qa_studies",
    ],
)
def test_w1469_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
