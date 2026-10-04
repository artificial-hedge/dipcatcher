import pytest


@pytest.mark.parametrize(
    "name",
    [
        "viperfish_qa_studies",
        "hatchetfish_qa_studies",
        "bristlemouth_qa_studies",
        "anglerfish_qa_studies",
        "grenadier_qa_studies",
        "lanternfish_qa_studies",
    ],
)
def test_w1618_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
