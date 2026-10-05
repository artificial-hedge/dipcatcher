import pytest


@pytest.mark.parametrize(
    "name",
    [
        "oriens_qa_studies",
        "vapula_qa_studies",
        "zagan_qa_studies",
        "valac_qa_studies",
        "flauros_qa_studies",
        "kimaris_qa_studies",
    ],
)
def test_w1954_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
