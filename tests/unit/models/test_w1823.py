import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tengri2_qa_studies",
        "ukerm2_qa_studies",
        "almas2_qa_studies",
        "khangai2_qa_studies",
        "sulde2_qa_studies",
        "shunu2_qa_studies",
    ],
)
def test_w1823_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
