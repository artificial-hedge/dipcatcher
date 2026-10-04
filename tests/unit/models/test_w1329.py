import pytest


@pytest.mark.parametrize(
    "name",
    [
        "looogle_studies",
        "marathon_eval_studies",
        "lost_middle_studies",
        "niah_v2_studies",
        "gov_report_studies",
        "passkey_retrieval_studies",
    ],
)
def test_w1329_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
