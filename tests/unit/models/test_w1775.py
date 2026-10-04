import pytest


@pytest.mark.parametrize(
    "name",
    [
        "inari_qa_studies",
        "kaguya_qa_studies",
        "momotaro_qa_studies",
        "urashima_qa_studies",
        "shichifukujin_qa_studies",
        "takemikazuchi_qa_studies",
    ],
)
def test_w1775_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
