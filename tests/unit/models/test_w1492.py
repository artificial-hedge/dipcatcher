import pytest


@pytest.mark.parametrize(
    "name",
    [
        "platypus_qa_studies",
        "bilby_qa_studies",
        "possum_qa_studies",
        "thylacine_qa_studies",
        "quoll_qa_studies",
        "echidna_qa_studies",
    ],
)
def test_w1492_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
