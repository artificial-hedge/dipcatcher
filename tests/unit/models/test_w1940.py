import pytest


@pytest.mark.parametrize(
    "name",
    [
        "thagya_min_qa_studies",
        "shwe_nabay_qa_studies",
        "taungmagyi_qa_studies",
        "magami_qa_studies",
        "mahagiri_qa_studies",
        "min_kyawzwa_qa_studies",
    ],
)
def test_w1940_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
