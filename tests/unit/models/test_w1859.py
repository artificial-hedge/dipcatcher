import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pwll_qa_studies",
        "manawydan_qa_studies",
        "llefelys_qa_studies",
        "cassivellaunus_qa_studies",
        "beli_qa_studies",
        "matholwch_qa_studies",
    ],
)
def test_w1859_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
