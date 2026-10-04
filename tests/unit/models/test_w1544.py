import pytest


@pytest.mark.parametrize(
    "name",
    [
        "toco_qa_studies",
        "nunlet_qa_studies",
        "nunbird_qa_studies",
        "puffbird_qa_studies",
        "hoopoe_qa_studies",
        "woodhoopoe_qa_studies",
    ],
)
def test_w1544_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
