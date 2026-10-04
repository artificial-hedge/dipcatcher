import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ea_qa_studies",
        "sargon_qa_studies",
        "semiramis_qa_studies",
        "utnapishtim_qa_studies",
        "humbaba_qa_studies",
        "pazuzu_qa_studies",
    ],
)
def test_w1786_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
