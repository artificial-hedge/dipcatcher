import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tinia_qa_studies",
        "menrva_qa_studies",
        "turan_qa_studies",
        "fufluns_qa_studies",
        "voltumna_qa_studies",
        "veltha_qa_studies",
    ],
)
def test_w1718_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
