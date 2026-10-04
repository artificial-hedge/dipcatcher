import pytest


@pytest.mark.parametrize(
    "name",
    [
        "titmouse_qa_studies",
        "tanager_qa_studies",
        "vireo_qa_studies",
        "grosbeak_qa_studies",
        "bunting_qa_studies",
        "nuthatch_qa_studies",
    ],
)
def test_w1509_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
