import pytest


@pytest.mark.parametrize(
    "name",
    [
        "jaguarundi_qa_studies",
        "ocelot_qa_studies",
        "margay_qa_studies",
        "serval_qa_studies",
        "puma_qa_studies",
        "caracal_qa_studies",
    ],
)
def test_w1476_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
