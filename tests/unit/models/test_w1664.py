import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kraken_qa_studies",
        "wyvern_qa_studies",
        "siren_qa_studies",
        "roc_qa_studies",
        "simurgh_qa_studies",
        "krampus_qa_studies",
    ],
)
def test_w1664_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
