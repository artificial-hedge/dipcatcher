import pytest


@pytest.mark.parametrize(
    "name",
    [
        "reo_qa_studies",
        "cosus_qa_studies",
        "quangeio_qa_studies",
        "aracus_qa_studies",
        "cronia_qa_studies",
        "munidis_qa_studies",
    ],
)
def test_w1858_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
