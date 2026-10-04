import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dangun_qa_studies",
        "hwanin_qa_studies",
        "hwanung_qa_studies",
        "samshin_qa_studies",
        "haenim_qa_studies",
        "dalnim_qa_studies",
    ],
)
def test_w1726_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
