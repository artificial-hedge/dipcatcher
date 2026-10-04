import pytest


@pytest.mark.parametrize(
    "name",
    [
        "taliesin_qa_studies",
        "arawn_qa_studies",
        "gwydion_qa_studies",
        "lludd_qa_studies",
        "lleu_qa_studies",
        "branwen_qa_studies",
    ],
)
def test_w1780_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
