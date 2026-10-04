import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chimpanzee_qa_studies",
        "bonobo_qa_studies",
        "douc_qa_studies",
        "siamang_qa_studies",
        "proboscis_qa_studies",
        "snub_nosed_qa_studies",
    ],
)
def test_w1594_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
