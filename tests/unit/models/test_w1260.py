import pytest


@pytest.mark.parametrize(
    "name",
    [
        "clinical_trial_studies",
        "adaptive_trial_studies",
        "meta_analysis_studies",
        "rwe_studies",
        "outcomes_research_studies",
        "comparative_effectiveness_studies",
    ],
)
def test_w1260_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
