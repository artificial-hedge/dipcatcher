import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hellaswag_studies",
        "boolq_studies",
        "piqa_studies",
        "siqa_studies",
        "openbookqa_studies",
        "copa_studies",
    ],
)
def test_w1304_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
