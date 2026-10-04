import pytest


@pytest.mark.parametrize(
    "name",
    [
        "amaterasu_qa_studies",
        "susanoo_qa_studies",
        "tsukuyomi_qa_studies",
        "inari_qa_studies",
        "hachiman_qa_studies",
        "raijin_qa_studies",
    ],
)
def test_w1699_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
