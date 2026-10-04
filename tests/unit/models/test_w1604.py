import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shrew_qa_studies",
        "mole_qa_studies",
        "gopher_qa_studies",
        "rabbit_qa_studies",
        "dassie_qa_studies",
        "springhare_qa_studies",
    ],
)
def test_w1604_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
