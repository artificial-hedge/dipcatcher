import pytest


@pytest.mark.parametrize(
    "name",
    [
        "steelhead_qa_studies",
        "grayling_qa_studies",
        "char_qa_studies",
        "dolly_varden_qa_studies",
        "whitefish_qa_studies",
        "sockeye_qa_studies",
    ],
)
def test_w1558_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
