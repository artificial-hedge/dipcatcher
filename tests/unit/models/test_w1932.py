import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ah_puch_qa_studies",
        "alux_qa_studies",
        "cizin_qa_studies",
        "xtabay_qa_studies",
        "vucub_qa_studies",
        "nahualli_qa_studies",
    ],
)
def test_w1932_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
