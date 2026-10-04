import pytest


@pytest.mark.parametrize(
    "name",
    [
        "watermark_studies",
        "sleepless_studies",
        "strip_defense_studies",
        "activation_cluster_studies",
        "fine_pruning_studies",
        "abs_scan_studies",
    ],
)
def test_w1313_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
