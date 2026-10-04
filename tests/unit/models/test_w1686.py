import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ninki_qa_studies",
        "nbanda_qa_studies",
        "ilomba_qa_studies",
        "abada_qa_studies",
        "adze_qa_studies",
        "okubi_qa_studies",
    ],
)
def test_w1686_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
