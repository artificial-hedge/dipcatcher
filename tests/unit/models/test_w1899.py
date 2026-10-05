import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pishacha_qa_studies",
        "daitya_qa_studies",
        "diti_qa_studies",
        "putana_qa_studies",
        "aghasura_qa_studies",
        "bakasura_qa_studies",
    ],
)
def test_w1899_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
