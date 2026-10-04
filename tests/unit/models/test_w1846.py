import pytest


@pytest.mark.parametrize(
    "name",
    [
        "almaqah_qa_studies",
        "athtar_qa_studies",
        "haubas_qa_studies",
        "aranyada_qa_studies",
        "nasr2_qa_studies",
        "anbay_qa_studies",
    ],
)
def test_w1846_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
