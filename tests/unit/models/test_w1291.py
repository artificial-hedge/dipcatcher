import pytest


@pytest.mark.parametrize(
    "name",
    [
        "data_mix_studies",
        "dedup_minhash_studies",
        "dedup_studies",
        "domain_classifier_studies",
        "perplexity_filter_studies",
        "quality_filter_studies",
    ],
)
def test_w1291_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
