import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bindus_qa_studies",
        "medaurus_qa_studies",
        "vidasus_qa_studies",
        "thana_qa_studies",
        "redon_qa_studies",
        "illyris_qa_studies",
    ],
)
def test_w1717_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
