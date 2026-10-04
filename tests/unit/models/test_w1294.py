import pytest


@pytest.mark.parametrize(
    "name",
    [
        "activation_oracle_studies",
        "concept_vector_studies",
        "feature_ablation_studies",
        "honesty_vector_studies",
        "reading_vector_studies",
        "refusal_vector_studies",
    ],
)
def test_w1294_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
