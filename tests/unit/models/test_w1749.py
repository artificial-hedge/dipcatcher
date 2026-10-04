import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chernobog_qa_studies",
        "belobog_qa_studies",
        "stribog_qa_studies",
        "hors_qa_studies",
        "dazhbog_qa_studies",
        "semargl_qa_studies",
    ],
)
def test_w1749_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
