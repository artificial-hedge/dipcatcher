import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mussel_qa_studies",
        "scallop_qa_studies",
        "conch_qa_studies",
        "clam_qa_studies",
        "oyster_qa_studies",
        "whelk_qa_studies",
    ],
)
def test_w1571_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
