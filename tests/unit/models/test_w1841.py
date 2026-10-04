import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chemosh3_qa_studies",
        "ashtar2_qa_studies",
        "baalpeor_qa_studies",
        "kiriath2_qa_studies",
        "dibon2_qa_studies",
        "nebo2_qa_studies",
    ],
)
def test_w1841_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
