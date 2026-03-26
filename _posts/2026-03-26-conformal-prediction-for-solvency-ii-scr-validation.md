---
layout: post
title: "Conformal Prediction for Solvency II SCR Validation"
date: 2026-03-26
categories: [pricing, techniques, research]
tags: [conformal-prediction, solvency-ii, scr, capital-modelling, model-validation, pra, ss1-24, var-backtesting, insurance-conformal, python, tweedie]
description: "Conformal prediction gives finite-sample valid 99.5% risk bounds for individual policies — useful for premium risk SCR validation and PRA SS1/24 backtesting, but not a replacement for aggregate capital modelling."
---

The SCR at 99.5% VaR has always had an uncomfortable relationship with the data. You need to estimate the 1-in-200 tail from a history that, for most personal lines books, contains perhaps 5-10 years of decent data. The standard tools — lognormal severity, Pareto tails, bootstrap CIs on GLM parameters — make distributional assumptions that are hard to validate in the tail and essentially impossible to challenge statistically with 200+ observations at the 99.5th percentile.

Conformal prediction is distribution-free. Its coverage guarantee — that the upper bound contains the true loss with at least 99.5% probability — holds without any assumptions about the underlying distribution, with finite samples, and for any predictor you care to use. That is genuinely different from what bootstrap or asymptotic theory offers.

The question for a pricing actuary is whether that theoretical advantage translates to anything useful for SCR work. We think it does in two specific places. We think it is useless, or actively misleading, in three others. The difference matters because conformal prediction is being talked about in capital modelling circles and the hype is running ahead of what the method actually delivers.

---

## What conformal prediction actually guarantees

Split conformal prediction works as follows. You fit a pricing model on training data. You hold back a calibration set. For each calibration observation, you compute a nonconformity score — for insurance losses, the Pearson residual `(y - ŷ) / ŷ^(p/2)` for Tweedie power `p` is the right choice (Manna et al., arXiv:2507.06921, July 2025). You take the (1-α) quantile of those calibration scores. At prediction time, you add that quantile back to produce an interval that is guaranteed to contain the true value with probability at least (1-α).

The guarantee is finite-sample: it holds for any sample size, not asymptotically. It is distribution-free: it requires exchangeability, not normality or any specific tail behaviour. And it is honest about what the predictor contributes: as Hong (arXiv:2601.21153, January 2026) shows via the h-transform framing, a better predictor produces narrower intervals but the coverage guarantee holds regardless of how bad your model is. This last point is important — it means a conformal 99.5% upper bound is a true 99.5% upper bound even if your GBM is substantially miscalibrated.

Hong (arXiv:2503.03659, March 2025) explicitly frames this within Solvency II: the finite-sample guarantee is the right standard for regulatory capital work, and asymptotic bootstrap CIs are not adequate substitutes. We agree with this argument.

---

## Where it helps: premium risk SCR validation

The standard approach to premium risk SCR in a UK personal lines internal model is to fit a parametric severity distribution (Pareto, lognormal, or generalised Pareto) and read off the 99.5th percentile. The parametric assumption carries model risk that is difficult to quantify. You can bootstrap, but bootstrap CIs on tail quantiles are notoriously slow to converge and rely on asymptotic normality of the quantile estimator.

Conformal prediction gives you something different: a per-risk upper bound at 99.5% with a finite-sample guarantee that doesn't depend on your distributional choice. Use it as a sanity check on your parametric SCR estimate, not as a replacement.

```python
from insurance_conformal import LocallyWeightedConformal
from insurance_conformal import SCRReport

# Fit a locally-weighted conformal predictor on your pricing model
lw = LocallyWeightedConformal(model=fitted_gbm, tweedie_power=1.5)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)

# SCR upper bounds per policy at alpha=0.005 (99.5%)
scr = SCRReport(predictor=lw, policy_ids=df_test["policy_id"])
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)

# scr_bounds columns: policy_id, expected_loss, upper_bound_99.5pct,
#                     scr_component, coverage_level
print(scr_bounds.head())
```

The `scr_component` column is `max(0, upper_bound - expected_loss)` per policy — the tail excess over the point forecast. For a portfolio where your parametric model gives a portfolio-level SCR of, say, 35% of expected loss, you should see the conformal per-risk components averaging in a similar range. If they are substantially higher, your parametric tail assumption is understating risk. If substantially lower, your Pareto or lognormal assumption is too heavy-tailed.

`LocallyWeightedConformal` is the right choice here over `InsuranceConformalPredictor` for SCR work because it uses a secondary GBM to model residual spread heterogeneity — the intervals are narrower (Manna et al. report approximately 24% reduction in interval width versus standard Pearson-weighted conformal) while maintaining the same coverage guarantee. Narrower intervals mean you are not overstating individual risk SCR components unnecessarily.

If you want the lightest-touch version — no secondary spread model, no assumption on your predictor quality — use `HongTransformConformal`. The h-transform approach from Hong (2026) gives a model-free 99.5% upper bound at the cost of wider intervals:

```python
from insurance_conformal import HongTransformConformal

ht = HongTransformConformal(model=fitted_glm)
ht.calibrate(X_cal, y_cal)

from insurance_conformal import solvency_capital_range
result = solvency_capital_range(ht, X_test, alpha=0.005,
                                exposure=df_test["years_on_cover"].to_numpy())
print(f"Total conformal SCR: {result.total_scr:,.0f}")
print(f"Mean interval width: {result.mean_interval_width:.2f}")
```

The `exposure` parameter handles policies with different term lengths — a 6-month policy contributes half as much as a full-year policy to `total_scr`.

---

## Where it helps: model validation and PRA SS1/24 backtesting

This is the application we find most compelling commercially, and it comes from Retzlaff et al. (ICML 2025), which proves formal equivalence between conformal coverage tests and standard VaR backtests — specifically the Kupiec (1995) and Christoffersen (1998) tests that market risk practitioners have used for decades.

PRA SS1/24 requires statistical testing of probability distribution forecasts. The standard Solvency II internal model validation tests include backtesting of the predicted loss distribution, but actuarial practice is inconsistent about which tests to apply and at what significance level. The Kupiec unconditional coverage test — which checks whether the observed exceedance rate matches the target — is equivalent to a conformal coverage test at the same alpha.

This matters because it provides a bridge to regulatory acceptance. A PRA validation team that understands Kupiec immediately understands what a conformal coverage table is telling them:

```python
# Run multi-alpha coverage validation on your holdout set
val_table = scr.coverage_validation_table(X_test, y_test,
                                           alphas=[0.005, 0.01, 0.05, 0.10])
print(scr.to_markdown())
```

The output is a table showing target coverage, empirical coverage, number covered, and whether the method meets the requirement at each alpha level. For a PRA submission, this is the backtesting evidence that your SCR model is not systematically understating tail risk.

The `coverage_validation_table` also accepts custom alpha lists, so you can match the specific tests in your ORSA and internal model validation framework. Run it across risk subgroups — vehicle age bands, NCD levels, geographic regions — using `subgroup_coverage` to check that the 99.5% guarantee holds for high-risk segments as well as on average:

```python
from insurance_conformal import subgroup_coverage

coverage_by_vehicle_age = subgroup_coverage(
    predictor=lw,
    X_test=X_test,
    y_test=y_test,
    groups=df_test["vehicle_age_band"],
    alpha=0.005,
)
```

Conformal coverage is a marginal guarantee — 99.5% on average across all policies. If your young-driver cohort achieves only 97% empirical coverage, the marginal guarantee is being met but the subgroup guarantee is not. SS1/24 requires testing at appropriate levels of granularity; this is how you do it.

---

## Where it does not help: aggregate portfolio SCR

We want to be direct about this because it is the obvious next question and the answer is disappointing.

Conformal prediction gives a 99.5% bound for each individual risk. To get a portfolio SCR, the obvious thing to do is sum the individual bounds. Do not do this. Summing individual 99.5% bounds gives you something like a 100% bound on the portfolio aggregate, because you are adding up rare events as if they all occur simultaneously. The resulting capital figure will be wildly conservative relative to any realistic dependence structure.

The `aggregate_scr()` method on `SCRReport` and the `total_scr` field on `SolvencyCapitalRange` give you the sum of per-risk conformal components. Both the docstrings and the source code explicitly warn that this does not account for dependence. We are flagging this again because we have seen the aggregate number quoted in context where it appears to be a portfolio SCR estimate — it is not.

For aggregate premium risk SCR, you need a copula or a factor model that captures the correlation structure across risks, followed by a simulation of portfolio aggregate losses. Conformal prediction does not help with this. The only conformal tool that makes contact with portfolio aggregation is `JointConformalPredictor` from `insurance_conformal.multivariate` — it gives joint bounds for frequency and severity simultaneously for a single risk, not across risks in a portfolio.

---

## Where it does not help: catastrophe risk SCR

The 99.5th percentile requires approximately 200 observations to estimate with any statistical reliability. Catastrophe risk SCR is calibrated from event catalogues containing decades of modelled event footprints, not from your observed claims history. You need the cat model vendors (RMS, AIR, Oasis) for this. Conformal prediction on cat claims is not sensible; you do not have 200 major flood or windstorm events in your UK motor or home book.

---

## Where it does not help (yet): reserve risk SCR

Reserve risk SCR in UK personal lines typically uses chain-ladder development factors with Mack uncertainty, or an overdispersed Poisson model. The uncertainty lives at the triangle level — how much does the total IBNR develop from here? This is a different object from an individual risk prediction interval.

You could theoretically apply conformal prediction to individual claim development — treat each open claim as an observation, fit a development model, and put conformal bounds on ultimate settlement per claim. Some of the infrastructure for this exists in `insurance_conformal.claims`. But we are not aware of published work that does this cleanly in a Solvency II reserve risk context, and the aggregation problem from portfolio SCR applies here too. We expect this to become an active research area; it is not solved.

---

## Premium sufficiency via conformal risk control

One application that is underappreciated is `PremiumSufficiencyController` from `insurance_conformal.risk`. This uses conformal risk control (Angelopoulos et al., ICLR 2024) to control expected loss ratios directly — not just coverage probability — at a specified threshold with a finite-sample guarantee.

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)
result = psc.predict(premium_new)
# result["upper_bound"] = loading factor that keeps E[loss ratio] <= alpha
```

This is meaningful for a pricing team trying to demonstrate premium adequacy for Solvency II purposes: you can show that the loading applied to your technical premium keeps the expected loss ratio below a given threshold with a controlled false positive rate. The distinction from pure coverage control is that you are controlling the loss function directly, not just whether the interval contains the observation.

---

## The regulatory position

EIOPA has not issued formal guidance on conformal methods as of early 2026. Present these outputs as supplementary technical validation — evidence alongside your parametric SCR estimates, not a replacement for them. The Retzlaff et al. ICML 2025 equivalence result is the best argument for PRA acceptability: if a PRA supervisor accepts Kupiec backtesting, they should accept a conformal coverage table, because they are mathematically the same test. But establishing this in practice will take time and depends on your relationship with your supervision team.

The `scr.py` module docstring notes this explicitly. We have not softened the language.

---

## The honest summary

| SCR component | Conformal value | Note |
|---|---|---|
| Premium risk — individual bounds | High | Finite-sample valid 99.5% upper bounds per policy. Use `LocallyWeightedConformal` + `SCRReport`. |
| Model validation / VaR backtesting | High | Formal equivalence to Kupiec/Christoffersen (Retzlaff et al. ICML 2025). Directly relevant to SS1/24. |
| Premium sufficiency | Moderate | `PremiumSufficiencyController` controls expected loss ratio, not just coverage. |
| Aggregate portfolio SCR | None | Summing individual bounds ignores diversification. Massively conservative. |
| Cat risk SCR | None | Needs 200+ observations at 99.5th percentile. Cat models are the right tool. |
| Reserve risk SCR | Open | Individual claim development conformal possible in principle; not published in a Solvency II context. |

The library is at [GitHub](https://github.com/burning-cost/insurance-conformal). The SCR-specific code lives in `scr.py` and `_solvency.py`; the validation workflow in `diagnostics.py` and `diagnostics_ext.py`.

```bash
uv add insurance-conformal
```

Key papers: Hong (arXiv:2503.03659), Hong (arXiv:2601.21153), Manna et al. (arXiv:2507.06921), Graziadei et al. (arXiv:2307.13124), Retzlaff et al. (ICML 2025).

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the core `insurance-conformal` library post: pearson_weighted scores, coverage-by-decile diagnostics, and the finite-sample coverage guarantee
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — the governance documentation layer: how to register a Solvency II conformal validation run in the MRM inventory
