import pytest


@pytest.mark.parametrize(
    "name",
    [
        "harrier_qa_studies",
        "kite_qa_studies",
        "kestrel_qa_studies",
        "osprey_qa_studies",
        "vulture_qa_studies",
        "condor_qa_studies",
    ],
)
def test_w1453_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
