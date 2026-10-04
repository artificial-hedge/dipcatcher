import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dromedary_qa_studies",
        "salt_qa_studies",
        "alpaca_qa_studies",
        "aoudad_qa_studies",
        "guanaco_qa_studies",
        "vicuna_qa_studies",
    ],
)
def test_w1606_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
