import pytest


@pytest.mark.parametrize(
    "name",
    [
        "albasty_qa_studies",
        "furts_qa_studies",
        "chinka_qa_studies",
        "albi_qa_studies",
        "gorgogosh_qa_studies",
        "rukhi_qa_studies",
    ],
)
def test_w1926_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
