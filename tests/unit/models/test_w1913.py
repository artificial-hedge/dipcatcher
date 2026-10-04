import pytest


@pytest.mark.parametrize(
    "name",
    [
        "navka_qa_studies",
        "berstuk_qa_studies",
        "rugievit_qa_studies",
        "chuma_qa_studies",
        "koshmar_qa_studies",
        "perekus_qa_studies",
    ],
)
def test_w1913_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
