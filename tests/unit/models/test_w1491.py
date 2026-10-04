import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cypress_qa_studies",
        "laurel_qa_studies",
        "hemlock_qa_studies",
        "magnolia_qa_studies",
        "eucalyptus_qa_studies",
        "spruce_qa_studies",
    ],
)
def test_w1491_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
