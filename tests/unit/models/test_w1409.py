import pytest


@pytest.mark.parametrize(
    "name",
    [
        "beerqa_lite_studies",
        "ensem_qa_studies",
        "cider_qa_studies",
        "fanqa_lite_studies",
        "hops_qa_studies",
        "bamboogle_lite_studies",
    ],
)
def test_w1409_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
