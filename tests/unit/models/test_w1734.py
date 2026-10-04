import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arawn_qa_studies",
        "gwydion_qa_studies",
        "llew_qa_studies",
        "rhiannon_qa_studies",
        "ceridwen_qa_studies",
        "taliesin_qa_studies",
    ],
)
def test_w1734_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
