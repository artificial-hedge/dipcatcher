import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ahura2_qa_studies",
        "verethragna2_qa_studies",
        "atar2_qa_studies",
        "haurvatat2_qa_studies",
        "ameretat2_qa_studies",
        "spenta2_qa_studies",
    ],
)
def test_w1806_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
