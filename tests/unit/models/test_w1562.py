import pytest


@pytest.mark.parametrize(
    "name",
    [
        "clownfish_qa_studies",
        "dragonet_qa_studies",
        "pufferfish_qa_studies",
        "mandarinfish_qa_studies",
        "boxfish_qa_studies",
        "pipefish_qa_studies",
    ],
)
def test_w1562_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
