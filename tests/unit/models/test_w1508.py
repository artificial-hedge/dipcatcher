import pytest


@pytest.mark.parametrize(
    "name",
    [
        "finch_qa_studies",
        "thrush_qa_studies",
        "sparrow_qa_studies",
        "wren_qa_studies",
        "chickadee_qa_studies",
        "warbler_qa_studies",
    ],
)
def test_w1508_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
