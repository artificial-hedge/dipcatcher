import pytest


@pytest.mark.parametrize(
    "name",
    [
        "daeva_qa_studies",
        "div_qa_studies",
        "fravashi_qa_studies",
        "khshathra_qa_studies",
        "pairika_qa_studies",
        "spenta_mainyu_qa_studies",
    ],
)
def test_w1890_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
