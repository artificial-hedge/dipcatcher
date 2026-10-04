import pytest


@pytest.mark.parametrize(
    "name",
    [
        "satyr_qa_studies",
        "mandarin_qa_studies",
        "moray_eel_qa_studies",
        "colugo_qa_studies",
        "geoffroy_qa_studies",
        "pangolin_2_qa_studies",
    ],
)
def test_w1628_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
