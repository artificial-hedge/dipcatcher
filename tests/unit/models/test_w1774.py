import pytest


@pytest.mark.parametrize(
    "name",
    [
        "thoth_qa_studies",
        "bastet_qa_studies",
        "sobek_qa_studies",
        "sekhmet_qa_studies",
        "hathor_qa_studies",
        "nut_qa_studies",
    ],
)
def test_w1774_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
