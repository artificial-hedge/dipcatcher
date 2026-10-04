import pytest


@pytest.mark.parametrize(
    "name",
    [
        "logiqa_log_studies",
        "recli_log_studies",
        "lsat_log_studies",
        "conseq_log_studies",
        "reason_mc_studies",
        "abductive_nli_studies",
    ],
)
def test_w1347_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
