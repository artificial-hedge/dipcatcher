import pytest


@pytest.mark.parametrize(
    "name",
    [
        "skink_qa_studies",
        "tuatara_qa_studies",
        "chameleon_qa_studies",
        "terrapin_qa_studies",
        "hognose_qa_studies",
        "anole_qa_studies",
    ],
)
def test_w1494_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
