---
layout: post
title: "Ogden Rate and PPOs: Pricing Large Bodily Injury in Python"
date: 2026-03-24
categories: [pricing, techniques, tutorials]
tags: [ogden-rate, ppo, bodily-injury, large-loss, reserving, discount-rate, lump-sum, motor-liability, employers-liability, uk-liability, python, insurance-conformal, insurance-governance, tutorial]
description: "How the Ogden discount rate and Periodical Payment Orders change the maths of large BI pricing in the UK — with Python code to calculate lump sum equivalents, discount PPO cash flows, and stress-test Ogden sensitivity."
---

Every UK motor liability and employers' liability pricing actuary knows the Ogden rate exists. Fewer have built the discount cash flow calculation from scratch, or thought carefully about what happens when the court awards a PPO instead of a lump sum. This post works through both.

The Ogden rate is currently -0.25% (set August 2019 under the Civil Liability Act 2018 framework). That negative sign is not a typo. On a serious injury claim where a 30-year-old claimant needs £60,000 per year in care costs for life, the lump sum under the current rate is over twice the lump sum at the 2.5% rate that applied before 2017. Get this wrong and the error sits quietly in your reserve until it does not.

---

## The Ogden rate: what it is and why it is negative

The Ogden discount rate is used by UK courts when awarding lump sum damages for future financial losses. The court multiplies the annual loss by an Ogden multiplier from the Government Actuary's Department tables to arrive at a lump sum. The multiplier depends on age, sex, assumed investment return (the Ogden rate), and mortality assumptions.

The logic is: a claimant receiving £X today can invest it and earn returns over the period, so the lump sum should be discounted by the assumed net rate of return. At a positive rate, the lump sum is less than the undiscounted sum of future losses. At -0.25%, the lump sum is slightly more.

The rate was set at 2.5% from 2001 to 2017. It then dropped to -0.75% in March 2017 — a shock to the industry that added billions to outstanding large BI reserves overnight. The Civil Liability Act 2018 changed the methodology: the rate is now set by reference to a diversified portfolio including equities (not purely index-linked gilts as previously), which produced the current -0.25% in August 2019. The next review was due in 2024; as of early 2027, no revision has been announced, though the Lord Chancellor retains the power to revise it.

The negative rate reflects the view that claimants investing a lump sum will achieve returns slightly below inflation — a cautious but not absurd assumption for a claimant who cannot bear investment risk. For insurers, it means every pound of future annual loss becomes slightly more than a pound of reserve.

---

## Computing lump sum equivalents in Python

The Ogden multiplier for a whole-life annuity is the present value of £1 per year, discounted at the Ogden rate, weighted by survival probabilities from the ONS or GAD life tables. The GAD tables are published at [gov.uk/actuaries-discount-rate](https://www.gov.uk/actuaries-discount-rate).

For illustration we use a simplified Makeham mortality law. The annuity factors below will differ from the published GAD tables — the GAD tables use a period life table with more granular age-specific mortality — but the sensitivity to rate changes is representative of the real structure.

```python
import numpy as np

# Ogden discount rates to compare
RATES = {
    "pre_2017":  0.025,   # 2.5%: rate before March 2017
    "post_2017": -0.0075, # -0.75%: March 2017 to August 2019
    "current":   -0.0025, # -0.25%: August 2019 onwards
    "stress_up": 0.010,   # +1.0%: plausible upward revision scenario
}

def ogden_annuity_factor(age: int, sex: str, rate: float, max_age: int = 110) -> float:
    """
    Whole-life Ogden annuity factor: PV of £1/year to a claimant aged `age`.
    Uses simplified Makeham mortality — illustrative only. Use GAD tables in production.
    """
    if sex == "M":
        alpha, beta, c = 0.0007, 0.00005, 1.095  # approximate UK male
    else:
        alpha, beta, c = 0.0004, 0.00003, 1.090  # approximate UK female

    factor = 0.0
    cumulative_survival = 1.0

    for t in range(max_age - age):
        curr_age = age + t
        # Makeham hazard
        mu = alpha + beta * (c ** curr_age)
        annual_survival = np.exp(-mu)
        factor += cumulative_survival * (1.0 / (1.0 + rate)) ** t
        cumulative_survival *= annual_survival

    return factor


# A 30-year-old female claimant requiring £60,000/year in care costs
claimant_age = 30
annual_loss = 60_000

print(f"{'Rate':<15} {'Annuity factor':>15} {'Lump sum':>15}")
print("-" * 47)
for label, rate in RATES.items():
    af = ogden_annuity_factor(age=claimant_age, sex="F", rate=rate)
    lump_sum = annual_loss * af
    print(f"{label:<15} {af:>15.2f} {lump_sum:>15,.0f}")
```

```
Rate              Annuity factor        Lump sum
-----------------------------------------------
pre_2017               29.99          1,799,636
post_2017              71.07          4,264,005
current                60.70          3,641,845
stress_up              42.66          2,559,613
```

The move from 2.5% to -0.75% in 2017 added £2.46m to this single claimant's award under these mortality assumptions. The current -0.25% sits at £3.64m. The stress scenario at +1% produces £2.56m — £1.08m less than today. Note: the simplified Makeham model overstates longevity for young claimants compared with the published GAD period life tables; the actual GAD multipliers are lower, but the sensitivity to rate changes is directionally identical.

On a portfolio of serious injury claims, this stress is not academic. A review that moves the rate from -0.25% to +0.5% would reduce large BI outstanding reserves by hundreds of thousands per open claim. That is why the rate review is a permanent agenda item at actuarial committees.

---

## Why PPOs change everything

A Periodical Payment Order bypasses the whole lump sum calculation. Instead of a single payment, the court awards an index-linked annuity paid directly by the insurer (or in practice, by a life office to which the insurer has ceded a structured settlement). The claimant receives, say, £60,000 per year (indexed to ASHE 6115, the earnings index for care workers) for life.

PPOs are available under section 2(1) of the Damages Act 1996 and have been in use since 2005. Courts must consider them for significant future loss, and the Court of Protection will often prefer them for claimants who lack capacity. In practice, the determining factor is whether the insurer can demonstrate to the court's satisfaction that it can meet the obligation — which for smaller insurers is a genuine constraint.

From a pricing perspective, PPOs shift the insurer's liability in three ways:

1. **Ogden rate irrelevance.** The lump sum formula with its discount rate disappears. The liability is now the present value of an indexed annuity, discounted at a rate the insurer chooses for reserving purposes (its asset return assumption). The Ogden rate sets the lump sum; the insurer's own discount rate sets the PPO reserve.

2. **Longevity risk transfers to the insurer.** A lump sum caps the insurer's exposure: if the claimant dies in year 5, the lump sum has already been paid. Under a PPO, the insurer pays until the claimant dies. Catastrophic injury cases are increasingly survivorship outliers: a 20-year-old who was expected to live to 72 may now live to 85 with improved care. That extra 13 years of care payments was not in the actuarial pricing.

3. **Earnings inflation mismatch.** Most PPOs index to ASHE 6115 (care worker wages), not CPI or RPI. Care worker wage inflation has consistently run above CPI — 4-7% per year in 2022-2024. Longer-term, ASHE 6115 has run 2-3% above CPI on average. Any insurer hedging PPO liabilities with CPI-linked bonds is carrying an unhedged basis risk.

---

## Valuing a PPO in Python

```python
def ppo_reserve(
    age: int,
    sex: str,
    annual_payment: float,
    insurer_discount_rate: float,
    wage_inflation: float,
    max_age: int = 110,
) -> float:
    """
    Present value of a PPO: indexed annuity discounted at the insurer's
    internal rate. Uses the same simplified Makeham mortality as above.
    In practice, use a full stochastic mortality model.
    """
    if sex == "M":
        alpha, beta, c = 0.0007, 0.00005, 1.095
    else:
        alpha, beta, c = 0.0004, 0.00003, 1.090

    reserve = 0.0
    cumulative_survival = 1.0

    for t in range(max_age - age):
        curr_age = age + t
        mu = alpha + beta * (c ** curr_age)
        annual_survival = np.exp(-mu)

        # Nominal payment grows with wage inflation; discounted at insurer rate
        indexed_payment = annual_payment * ((1 + wage_inflation) ** t)
        discount_factor = (1 / (1 + insurer_discount_rate)) ** t
        reserve += cumulative_survival * indexed_payment * discount_factor

        cumulative_survival *= annual_survival

    return reserve


# The same 30-year-old female claimant; annual payment £60,000 indexed to ASHE 6115
claimant_age = 30
annual_payment = 60_000
ogden_lump_sum = annual_payment * ogden_annuity_factor(claimant_age, "F", -0.0025)

scenarios = [
    ("Base: 4% disc / 3% wage", 0.04, 0.03),
    ("High wage inflation",       0.04, 0.05),
    ("Low rate / high wage",      0.02, 0.05),
    ("High rate / low wage",      0.06, 0.02),
]

print(f"PPO reserve sensitivity — 30F, £60,000/year indexed to ASHE 6115")
print(f"Ogden lump sum (current rate): £{ogden_lump_sum:,.0f}")
print()
print(f"{'Scenario':<30} {'Reserve':>12} {'vs Ogden lump sum':>20}")
print("-" * 64)
for label, disc_r, wage_inf in scenarios:
    r = ppo_reserve(claimant_age, "F", annual_payment, disc_r, wage_inf)
    diff = r - ogden_lump_sum
    print(f"{label:<30} {r:>12,.0f} {diff:>+20,.0f}")
```

```
PPO reserve sensitivity — 30F, £60,000/year indexed to ASHE 6115
Ogden lump sum (current rate): £3,641,845

Scenario                            Reserve   vs Ogden lump sum
----------------------------------------------------------------
Base: 4% disc / 3% wage           2,579,067          -1,062,778
High wage inflation                4,557,987            +916,142
Low rate / high wage               9,273,957          +5,632,113
High rate / low wage               1,368,600          -2,273,245
```

The spread across scenarios is enormous. The base case at 4% discount and 3% wage inflation produces a PPO reserve £1m below the Ogden lump sum — the PPO looks favourable for the insurer on this basis. But the low-rate/high-wage scenario produces a reserve of £9.3m, nearly £5.6m above the lump sum. The high-wage-inflation scenario alone at £4.6m exceeds the Ogden lump sum by £916k.

The practical implication: an insurer that reserves PPOs at the base discount assumption and then faces a decade of ASHE 6115 inflation running above 3% will find its PPO reserves systematically inadequate. The scenario table is the governance artefact that belongs in the reserving committee pack.

---

## What this means for the loss distribution

Traditional large BI pricing assumes a lump sum distribution: severity is a single draw from a heavy-tailed distribution, typically Pareto or Burr, with the Ogden rate embedded in the expected severity at the time of settlement. The loss distribution has a definite tail — fat, but bounded for any individual claim.

PPOs change the shape. Instead of a single large payment, the insurer has a stream of payments lasting potentially 50+ years. The duration risk is not captured in the original loss distribution at all. An insurer that prices large BI using a historic severity distribution built on lump sum settlements is systematically under-counting the risk on claims that eventually become PPOs.

The propensity of a claim to convert to a PPO is hard to model. The main drivers are:

- **Claim size.** Claims above approximately £500k are consistently more likely to attract a PPO application. Below £250k, PPOs are rare.
- **Claimant solicitor strategy.** A well-resourced claimant legal team will push for a PPO if the current Ogden rate produces a lump sum that looks conservative compared with a PPO valued at the insurer's assumed discount rate. At -0.25%, the incentive to prefer a lump sum is lower than it was at -0.75%, but the analysis is claim-specific.
- **Court appetite.** Different circuit judges have materially different propensities. There is no systematic public data on this.
- **Insurer financial strength.** Courts will not impose a PPO on an insurer that cannot demonstrate capacity to meet the obligation indefinitely. Lloyd's syndicates, for example, have historically had more PPOs directed to structured settlement providers rather than left on the syndicate.

None of this is cleanly modellable from policy data. It is a judgment call, and should be documented as one.

---

## Where insurance-conformal fits in

The large BI severity distribution is genuinely fat-tailed and highly uncertain. The 95th-percentile severity on a commercial motor or employers' liability claim is not well-estimated from a point prediction alone, and the distributional assumptions (Pareto shape, Burr parameters) are fitted on thin data by definition — serious injury claims above £500k are rare.

[`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) produces distribution-free prediction intervals around severity estimates. For large BI, the relevant output is the upper bound of the interval: what is the plausible range of the claim, not just the point estimate? Applied to a large BI portfolio, the conformal interval width by claim severity decile gives you a direct view of where the model's tail uncertainty is largest.

The non-conformity score for severity models in this range is the log-residual: `|log(y) − log(ŷ)|`. Pearson-weighted scoring (the default for Tweedie models) is less appropriate when the tail is genuinely Pareto-like rather than variance-scaled. The `nonconformity="log_residual"` option is the right choice for large BI severity.

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=severity_model,
    nonconformity="log_residual",  # appropriate for heavy-tailed severity
)
cp.calibrate(X_cal, y_cal)

intervals = cp.predict_interval(X_large_bi, alpha=0.10)  # 90% intervals
# DataFrame columns: lower, point, upper

# Flag claims where the upper bound is above the PPO threshold
intervals["ppo_flag"] = intervals["upper"] > 500_000
```

The `ppo_flag` column identifies claims where, conditional on the model's point estimate and the calibrated uncertainty, a PPO outcome is plausible. These claims deserve different reserving treatment: not a single lump sum reserve, but a scenario reserve that spans both the lump sum and the adverse PPO wage inflation case.

---

## Where insurance-governance fits in

Large BI models are high-stakes: a systematic error in the Ogden rate assumption or the PPO propensity estimate will affect reserves, capital, and reinsurance pricing. [`insurance-governance`](https://github.com/burning-cost/insurance-governance) generates validation reports that surface sensitivity tests as explicit, auditable outputs.

The Ogden rate sensitivity — the difference in reserves between the current rate, the pre-2017 rate, and stress scenarios — should appear in every large BI model's validation report. Without it, the model risk committee has no basis for assessing whether the model is adequately reserved against a rate change.

```python
from insurance_governance import ModelValidationReport, ValidationModelCard
import numpy as np

card = ValidationModelCard(
    name="Large BI Severity v2.1",
    version="2.1.0",
    purpose="Large bodily injury severity estimation for UK motor and EL",
    methodology="Burr distribution with GBM covariates; lump sum basis",
    target="claim_severity",
    features=["age", "injury_type", "liability_split", "legal_rep"],
    limitations=[
        "Lump sum basis only — PPO propensity not modelled",
        "Ogden rate hard-coded at -0.25%; no dynamic rate input",
        "Calibrated on 2015-2023 settlements; pre-dates current claims environment",
    ],
    owner="Reserving & Pricing",
)

report = ModelValidationReport(
    model_card=card,
    y_val=y_large_bi_val,
    y_pred_val=preds_val,
    exposure_val=np.ones(len(y_large_bi_val)),
)
report.run()
report.to_html("large_bi_severity_validation.html")
```

The `limitations` field is where the Ogden and PPO assumptions belong. PRA supervisory visits to large motor insurers have increasingly focused on whether large BI reserving models have documented their Ogden rate assumptions explicitly — not as a footnote, but as a named limitation with a stated sensitivity range. A governance report that shows the model is hard-coded to the current rate, with no mechanism to update it when the rate changes, is a finding.

---

## What you cannot model

Two things are genuinely not modellable and should not be dressed up as if they were.

**Ogden rate changes are political and judicial, not statistical.** The rate is set by the Lord Chancellor after actuarial advice from the GAD. The process is rule-based (the Civil Liability Act 2018 specifies the methodology) but the output depends on current gilt yields, equity risk premium assumptions, and the political context at the time of review. You can build a scenario grid — as we have above — but do not attach a probability distribution to the scenarios. Any model that says "we estimate a 35% probability the Ogden rate moves to +0.5% by 2026" is fabricating precision.

**PPO propensity is driven by claimant solicitor strategy, not features of the risk.** The decision to push for a PPO is made by the claimant's legal team. A claims feature model built on policy data will capture some correlation — claim size, injury type, age — but will miss the dominant variable, which is which law firm is on the other side of the claim. We do not have that data at pricing time, and modelling around it produces false confidence.

Both limitations belong in the governance documentation. Uncertainty is not a problem to solve before presenting to the model risk committee; it is information the committee needs.

---

## Summary

The Ogden rate at -0.25% produces lump sum awards that are roughly double those calculated at the pre-2017 rate of 2.5%, for a young serious injury claimant with substantial ongoing care needs. PPOs remove the Ogden calculation entirely and replace it with a duration and wage inflation problem that is harder to hedge and harder to reserve for.

The Python code above gets you to a defensible lump sum reserve and a PPO sensitivity table. Use `insurance-conformal` to put uncertainty bounds on individual claim severity estimates — the `log_residual` non-conformity score is appropriate for this distribution. Use `insurance-governance` to document the Ogden rate assumption, the PPO propensity assumption, and their sensitivities in the model validation report where they belong.

What you cannot do is predict when the rate will change or how many open claims will convert to PPOs. Document that honestly.

```bash
uv add insurance-conformal
uv add insurance-governance
```

Sources: [GAD Ogden tables](https://www.gov.uk/actuaries-discount-rate) — [Civil Liability Act 2018](https://www.legislation.gov.uk/ukpga/2018/29/contents) — [Damages Act 1996, s.2](https://www.legislation.gov.uk/ukpga/1996/48/section/2) — [ASHE 6115 (ONS)](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours).

---

**Related posts:**

- [How to Build a Large Loss Loading Model for Home Insurance](/2026/10/14/large-loss-loading-for-home-insurance/) — quantile GBM approach to per-risk large loss loadings; the tail structure is directly applicable to large BI severity
- [Does Conformal Prediction Actually Work for Insurance Claims?](/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/) — the benchmark that shows why parametric intervals fail in the tail for heterogeneous books
- [Your Model Risk Register Is a Spreadsheet](/2026/03/13/your-model-risk-register-is-a-spreadsheet/) — why governance documentation for high-stakes models needs more than an Excel tab
