import pytest


@pytest.mark.parametrize(
    "name",
    [
        "roach_qa_studies",
        "barbel_qa_studies",
        "tench_qa_studies",
        "carp_qa_studies",
        "bream_qa_studies",
        "minnow_qa_studies",
    ],
)
def test_w1567_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
