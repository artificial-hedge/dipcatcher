import pytest


@pytest.mark.parametrize(
    "name",
    [
        "apedemak_qa_studies",
        "sebiumeker_qa_studies",
        "mandulis_qa_studies",
        "arensnuphis_qa_studies",
        "dedwen_qa_studies",
        "miket_qa_studies",
    ],
)
def test_w1848_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
