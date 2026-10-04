import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sobek_qa_studies",
        "anubis_qa_studies",
        "bastet_qa_studies",
        "neith_qa_studies",
        "khonsu_qa_studies",
        "min_qa_studies",
    ],
)
def test_w1751_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
