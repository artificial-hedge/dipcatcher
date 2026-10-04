import pytest


@pytest.mark.parametrize(
    "name",
    [
        "domovoi_qa_studies",
        "rusalka_qa_studies",
        "vodianoi_qa_studies",
        "leshy_qa_studies",
        "kikimora_qa_studies",
        "polevik_qa_studies",
    ],
)
def test_w1672_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
