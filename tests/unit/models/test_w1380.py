import pytest


@pytest.mark.parametrize(
    "name",
    [
        "art_nli_studies",
        "snips_lite_studies",
        "recast_lite_studies",
        "social_lite_studies",
        "subj_lite_studies",
        "para_paws_studies",
    ],
)
def test_w1380_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
