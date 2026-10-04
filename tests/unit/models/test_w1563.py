import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mantis_shrimp_qa_studies",
        "hermit_crab_qa_studies",
        "decorator_crab_qa_studies",
        "porcelain_crab_qa_studies",
        "pistol_shrimp_qa_studies",
        "cleaner_shrimp_qa_studies",
    ],
)
def test_w1563_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
