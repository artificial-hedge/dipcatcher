import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sst2_lite_studies",
        "cola_lite_studies",
        "qnli_lite_studies",
        "qqp_lite_studies",
        "stsb_lite_studies",
        "wnli_lite_studies",
    ],
)
def test_w1349_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
