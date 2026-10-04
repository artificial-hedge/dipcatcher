import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ognios_qa_studies",
        "nantosuelta_qa_studies",
        "braciaca_qa_studies",
        "antenociticus_qa_studies",
        "ares_lusitani_qa_studies",
        "deiba_qa_studies",
    ],
)
def test_w1867_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
