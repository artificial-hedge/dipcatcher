import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shams_qa_studies",
        "raymah_qa_studies",
        "hawl_qa_studies",
        "khalasah_qa_studies",
        "dhatanwat_qa_studies",
        "dhatzahran_qa_studies",
    ],
)
def test_w1847_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
