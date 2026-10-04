import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cedar_qa_studies",
        "maple_qa_studies",
        "elm_qa_studies",
        "oak_qa_studies",
        "willow_qa_studies",
        "birch_qa_studies",
    ],
)
def test_w1445_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
