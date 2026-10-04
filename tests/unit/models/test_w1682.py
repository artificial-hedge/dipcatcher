import pytest


@pytest.mark.parametrize(
    "name",
    [
        "veles_qa_studies",
        "zhaba_qa_studies",
        "sirin_qa_studies",
        "alkonost_qa_studies",
        "gamayun_qa_studies",
        "zmei_qa_studies",
    ],
)
def test_w1682_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
