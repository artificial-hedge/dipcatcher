import pytest


@pytest.mark.parametrize(
    "name",
    [
        "imagenet_a_studies",
        "imagenet_o_studies",
        "imagenet_v2_studies",
        "imagenet_e_studies",
        "imagenet_sketch_studies",
        "stylized_studies",
    ],
)
def test_w1315_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
