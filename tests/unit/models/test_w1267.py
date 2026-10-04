import pytest


@pytest.mark.parametrize(
    "name",
    [
        "alpha_tensor_studies",
        "differentiable_sat_studies",
        "neural_theorem_studies",
        "program_synthesis_studies",
        "sketch_programming_studies",
        "symbolic_regression_dl_studies",
    ],
)
def test_w1267_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
