import pytest


@pytest.mark.parametrize(
    "name",
    [
        "viracocha_qa_studies",
        "inti_qa_studies",
        "pachacamac_qa_studies",
        "supay_qa_studies",
        "guanare_qa_studies",
        "coniraya_qa_studies",
    ],
)
def test_w1767_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
