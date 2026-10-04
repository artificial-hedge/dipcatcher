import pytest


@pytest.mark.parametrize(
    "name",
    [
        "indiges_qa_studies",
        "lar_qa_studies",
        "penates_qa_studies",
        "numen_qa_studies",
        "terminus_qa_studies",
        "vertumnus_qa_studies",
    ],
)
def test_w1688_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
