import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dievas_qa_studies",
        "saulute_qa_studies",
        "medeina_qa_studies",
        "velnias_qa_studies",
        "ragana_qa_studies",
        "gabija_qa_studies",
    ],
)
def test_w1735_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
