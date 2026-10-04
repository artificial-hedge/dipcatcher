import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nuckelavee_qa_studies",
        "each_uisge_qa_studies",
        "mooinjer_qa_studies",
        "cabyll_qa_studies",
        "bugul_noz_qa_studies",
        "morveren_qa_studies",
    ],
)
def test_w1872_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
