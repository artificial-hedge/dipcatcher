import pytest


@pytest.mark.parametrize(
    "name",
    [
        "best_of_n_studies",
        "cdpo_studies",
        "constitutional_ai_studies",
        "orpo_studies",
        "simpo_studies",
        "sppo_studies",
    ],
)
def test_w1269_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
