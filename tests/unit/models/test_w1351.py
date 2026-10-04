import pytest


@pytest.mark.parametrize(
    "name",
    [
        "moral_found_studies",
        "ethic_jiminy_studies",
        "scruples_lite_studies",
        "moral_exc_studies",
        "principlism_toy_studies",
        "virtue_ethics_studies",
    ],
)
def test_w1351_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
