import pytest


@pytest.mark.parametrize(
    "name",
    [
        "airtasks_studies",
        "mmind2web_studies",
        "maze_eval_studies",
        "screenqa_studies",
        "weblinx_studies",
        "browsergym_studies",
    ],
)
def test_w1385_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
