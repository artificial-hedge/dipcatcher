import pytest


@pytest.mark.parametrize(
    "name",
    [
        "attribution_patching_studies",
        "causal_scrubbing_studies",
        "function_vector_studies",
        "induction_head_studies",
        "monosemantic_studies",
        "superposition_studies",
    ],
)
def test_w1276_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
