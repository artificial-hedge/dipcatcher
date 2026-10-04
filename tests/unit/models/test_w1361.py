import pytest


@pytest.mark.parametrize(
    "name",
    [
        "evidence_inf_studies",
        "scidtb_lite_studies",
        "hoax_detect_studies",
        "liar_lite_studies",
        "rumor_eval_studies",
        "covid_lies_studies",
    ],
)
def test_w1361_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
