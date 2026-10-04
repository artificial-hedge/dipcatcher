import pytest


@pytest.mark.parametrize(
    "name",
    [
        "backdoor_studies",
        "trojan_studies",
        "data_poison_studies",
        "clean_label_studies",
        "neural_cleanse_studies",
        "spectral_signature_studies",
    ],
)
def test_w1301_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
