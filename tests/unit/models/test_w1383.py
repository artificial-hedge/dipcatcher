import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aime24_studies",
        "mmmlu_lite_studies",
        "hle_lite_studies",
        "olympiadbench_studies",
        "super_gpqa_studies",
        "gpqa_diamond_studies",
    ],
)
def test_w1383_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
