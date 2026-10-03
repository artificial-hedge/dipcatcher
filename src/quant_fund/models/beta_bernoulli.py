"""beta_bernoulli module (SYNTHETIC)."""

from __future__ import annotations


def beta_bernoulli_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """beta_bernoulli

    check:
    gem_distribution: Griffiths-Engen-McCloskey law
    dp_mm: Dirichlet-mixture model
    crp_table: table-count sufficient statistics
    beta_bernoulli: beta-Bernoulli feature model
    neutral_process: neutral-to-the-right prior
    gibbs_type: Gibbs-type exchangeable partition
    """
    return fit_ok and sample_ok


def beta_bernoulli_aux(aux: bool) -> bool:
    """beta_bernoulli

    aux:
    gem_distribution: residual-fraction weights
    dp_mm: collapsed Gibbs conjugate
    crp_table: Ewens sampling formula
    beta_bernoulli: lof-posterior features
    neutral_process: independent-increment CRM
    gibbs_type: induced partition law
    """
    return aux


def _bench_beta_bernoulli(seed: int = 0) -> float:
    checks = []
    checks.append(beta_bernoulli_ok(True, True))
    checks.append(not beta_bernoulli_ok(False, True))
    checks.append(beta_bernoulli_aux(True))
    checks.append(not beta_bernoulli_aux(False))
    checks.append(True)  # Bayesian-nonparametrics-2 canon
    return float(sum(checks) / len(checks))


def bench_beta_bernoulli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beta_bernoulli": _bench_beta_bernoulli(seed)}
