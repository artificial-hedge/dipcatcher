import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cue_conflict_studies",
        "geirhos_studies",
        "texture_bias_studies",
        "shape_bias_studies",
        "imagenet_bg_studies",
        "backgrounds_studies",
    ],
)
def test_w1316_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
