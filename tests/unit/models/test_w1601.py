import pytest


@pytest.mark.parametrize(
    "name",
    [
        "free_tailed_qa_studies",
        "mouse_eared_qa_studies",
        "bulldog_bat_qa_studies",
        "fruit_bat_qa_studies",
        "tent_bat_qa_studies",
        "blossom_bat_qa_studies",
    ],
)
def test_w1601_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
