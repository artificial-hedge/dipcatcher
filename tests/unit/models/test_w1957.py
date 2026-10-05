import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bifrons_qa_studies",
        "vuall_qa_studies",
        "haagenti_qa_studies",
        "crocell_qa_studies",
        "bael_qa_studies",
        "gamigin_qa_studies",
    ],
)
def test_w1957_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
