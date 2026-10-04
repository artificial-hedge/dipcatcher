import pytest


@pytest.mark.parametrize(
    "name",
    [
        "samshin_qa_studies",
        "shimchong_qa_studies",
        "kongjwi_qa_studies",
        "bari_qa_studies",
        "ondal_qa_studies",
        "pyonggang_qa_studies",
    ],
)
def test_w1776_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
