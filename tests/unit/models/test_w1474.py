import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cormorant_qa_studies",
        "ibis_qa_studies",
        "curlew_qa_studies",
        "kingfisher_qa_studies",
        "loon_qa_studies",
        "bittern_qa_studies",
    ],
)
def test_w1474_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
