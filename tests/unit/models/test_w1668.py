import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ulfhednar_qa_studies",
        "berserkr_qa_studies",
        "vargr_qa_studies",
        "fafnir_qa_studies",
        "regin_qa_studies",
        "jotun_qa_studies",
    ],
)
def test_w1668_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
