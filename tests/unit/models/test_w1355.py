import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arc_hard2_studies",
        "piqa_lite_studies",
        "csqa_lite_studies",
        "swag_lite_studies",
        "hellaswag_lite_studies",
        "prost_lite_studies",
    ],
)
def test_w1355_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
