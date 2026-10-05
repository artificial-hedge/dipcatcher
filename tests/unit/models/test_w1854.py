import pytest


@pytest.mark.parametrize(
    "name",
    [
        "baal_hammon_qa_studies",
        "tanit_punic_qa_studies",
        "mekal_qa_studies",
        "sid_qa_studies",
        "yamm_qa_studies",
        "astarte_punic_qa_studies",
    ],
)
def test_w1854_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
