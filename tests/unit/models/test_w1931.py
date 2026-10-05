import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pukwudgie_qa_studies",
        "manitou_qa_studies",
        "naagloshii_qa_studies",
        "kachina_qa_studies",
        "deer_woman_qa_studies",
        "mishipeshu_qa_studies",
    ],
)
def test_w1931_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
