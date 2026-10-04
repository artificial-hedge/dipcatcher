import pytest


@pytest.mark.parametrize(
    "name",
    [
        "selkie_qa_studies",
        "kelpie_qa_studies",
        "puca_qa_studies",
        "banshee_qa_studies",
        "leprechaun_qa_studies",
        "dullahan_qa_studies",
    ],
)
def test_w1662_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
