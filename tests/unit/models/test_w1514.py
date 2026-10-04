import pytest


@pytest.mark.parametrize(
    "name",
    [
        "swallowtail_qa_studies",
        "cabbage_white_qa_studies",
        "fritillary_qa_studies",
        "painted_lady_qa_studies",
        "blue_morpho_qa_studies",
        "monarch_qa_studies",
    ],
)
def test_w1514_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
