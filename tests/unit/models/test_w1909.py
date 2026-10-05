import pytest


@pytest.mark.parametrize(
    "name",
    [
        "azhi_dahaka_qa_studies",
        "nasu_qa_studies",
        "druj_qa_studies",
        "jahi_qa_studies",
        "aeshma_qa_studies",
        "astwihad_qa_studies",
    ],
)
def test_w1909_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
