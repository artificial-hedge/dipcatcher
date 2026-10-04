import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hulder_qa_studies",
        "draugar_qa_studies",
        "alfar_qa_studies",
        "svartalf_qa_studies",
        "muspell_qa_studies",
        "ymir_qa_studies",
    ],
)
def test_w1692_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
