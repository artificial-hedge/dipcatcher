import pytest


@pytest.mark.parametrize(
    "name",
    [
        "palm_qa_studies",
        "acacia_qa_studies",
        "baobab_qa_studies",
        "sycamore_qa_studies",
        "alder_qa_studies",
        "olive_qa_studies",
    ],
)
def test_w1501_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
