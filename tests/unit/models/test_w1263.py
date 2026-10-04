import pytest


@pytest.mark.parametrize(
    "name",
    [
        "diagnostic_meta_studies",
        "fragility_index_studies",
        "individual_patient_meta_studies",
        "network_meta_studies",
        "trial_sequential_studies",
        "umbrella_review_studies",
    ],
)
def test_w1263_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
