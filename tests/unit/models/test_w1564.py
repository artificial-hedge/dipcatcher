import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cuttlefish_qa_studies",
        "nautilus_qa_studies",
        "nudibranch_qa_studies",
        "vampire_squid_qa_studies",
        "bobtail_squid_qa_studies",
        "sea_slug_qa_studies",
    ],
)
def test_w1564_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
