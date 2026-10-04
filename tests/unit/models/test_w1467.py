import pytest


@pytest.mark.parametrize(
    "name",
    [
        "steppe_qa_studies",
        "tundra_qa_studies",
        "summit_qa_studies",
        "valley_qa_studies",
        "volcano_qa_studies",
        "arch_qa_studies",
    ],
)
def test_w1467_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
