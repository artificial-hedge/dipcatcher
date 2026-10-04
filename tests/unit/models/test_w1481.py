import pytest


@pytest.mark.parametrize(
    "name",
    [
        "katydid_qa_studies",
        "weevil_qa_studies",
        "wasp_qa_studies",
        "mayfly_qa_studies",
        "stonefly_qa_studies",
        "earwig_qa_studies",
    ],
)
def test_w1481_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
