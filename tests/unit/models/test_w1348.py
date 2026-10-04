import pytest


@pytest.mark.parametrize(
    "name",
    [
        "anli_r1_studies",
        "anli_r2_studies",
        "anli_r3_studies",
        "mnli_match_studies",
        "snli_lite_studies",
        "scitail_lite_studies",
    ],
)
def test_w1348_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
