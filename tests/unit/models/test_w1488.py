import pytest


@pytest.mark.parametrize(
    "name",
    [
        "panther_qa_studies",
        "bobcat_qa_studies",
        "dingo_qa_studies",
        "kodkod_qa_studies",
        "oncilla_qa_studies",
        "tiger_qa_studies",
    ],
)
def test_w1488_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
