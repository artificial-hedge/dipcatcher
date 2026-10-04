from quant_fund.models.dual_ab_var import bench_dual_ab_var
from quant_fund.models.etale_cover import bench_etale_cover
from quant_fund.models.hom_stack_toy import bench_hom_stack_toy
from quant_fund.models.jacobian_toy import bench_jacobian_toy
from quant_fund.models.picard_variety import bench_picard_variety
from quant_fund.models.seesaw_theorem import bench_seesaw_theorem


def test_etale_cover():
    assert bench_etale_cover()["synthetic_etale_cover"] == 1.0


def test_jacobian_toy():
    assert bench_jacobian_toy()["synthetic_jacobian_toy"] == 1.0


def test_hom_stack_toy():
    assert bench_hom_stack_toy()["synthetic_hom_stack_toy"] == 1.0


def test_seesaw_theorem():
    assert bench_seesaw_theorem()["synthetic_seesaw_theorem"] == 1.0


def test_picard_variety():
    assert bench_picard_variety()["synthetic_picard_variety"] == 1.0


def test_dual_ab_var():
    assert bench_dual_ab_var()["synthetic_dual_ab_var"] == 1.0
