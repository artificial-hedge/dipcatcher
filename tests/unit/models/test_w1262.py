import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dynamic_borrowing_studies",
        "e_value_studies",
        "master_protocol_studies",
        "stepped_wedge_studies",
        "target_trial_emulation_studies",
        "win_ratio_studies",
    ],
)
def test_w1262_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
