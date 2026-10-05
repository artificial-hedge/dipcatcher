import pytest


@pytest.mark.parametrize(
    "name",
    [
        "thoth2_qa_studies",
        "nephthys2_qa_studies",
        "bes2_qa_studies",
        "sekhmet2_qa_studies",
        "ptah2_qa_studies",
        "geb2_qa_studies",
    ],
)
def test_w1811_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
