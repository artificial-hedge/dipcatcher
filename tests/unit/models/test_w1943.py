import pytest


@pytest.mark.parametrize(
    "name",
    [
        "leyak_qa_studies",
        "pocong_qa_studies",
        "kuntilanak_qa_studies",
        "genderuwo_qa_studies",
        "jenglot_qa_studies",
        "tuyul_qa_studies",
    ],
)
def test_w1943_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
