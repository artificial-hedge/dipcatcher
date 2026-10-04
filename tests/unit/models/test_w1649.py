import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bakunawa_qa_studies",
        "tikbalang_qa_studies",
        "kapre_qa_studies",
        "berbalang_qa_studies",
        "aswang_qa_studies",
        "sigbin_qa_studies",
    ],
)
def test_w1649_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
