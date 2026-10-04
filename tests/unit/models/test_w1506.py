import pytest


@pytest.mark.parametrize(
    "name",
    [
        "eider_qa_studies",
        "merganser_qa_studies",
        "scoter_qa_studies",
        "bufflehead_qa_studies",
        "canvasback_qa_studies",
        "mallard_qa_studies",
    ],
)
def test_w1506_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
