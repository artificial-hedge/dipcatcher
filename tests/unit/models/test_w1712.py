import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vahagn_qa_studies",
        "nahapet_qa_studies",
        "tir_qa_studies",
        "hayk_qa_studies",
        "astghik_qa_studies",
        "nane_qa_studies",
    ],
)
def test_w1712_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
