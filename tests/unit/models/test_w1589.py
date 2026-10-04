import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vaquita_qa_studies",
        "river_dolphin_qa_studies",
        "spinner_qa_studies",
        "right_whale_qa_studies",
        "rissos_qa_studies",
        "porpoise_qa_studies",
    ],
)
def test_w1589_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
