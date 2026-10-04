import pytest


@pytest.mark.parametrize(
    "name",
    [
        "swallow_qa_studies",
        "swiftlet_qa_studies",
        "martin_qa_studies",
        "needletail_qa_studies",
        "treeswift_qa_studies",
        "swift_qa_studies",
    ],
)
def test_w1535_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
