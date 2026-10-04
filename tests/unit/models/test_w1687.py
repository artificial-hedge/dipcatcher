import pytest


@pytest.mark.parametrize(
    "name",
    [
        "apep_qa_studies",
        "sobek_qa_studies",
        "bastet_qa_studies",
        "khonsu_qa_studies",
        "akh_qa_studies",
        "abti_qa_studies",
    ],
)
def test_w1687_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
