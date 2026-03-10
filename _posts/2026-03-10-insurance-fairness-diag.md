---
layout: post
title: "Your Pricing Model Is Discriminating. Here's Which Factor Is Doing It."
date: 2026-03-10
categories: [libraries, compliance]
tags: [fairness, proxy-discrimination, D-proxy, Shapley, FCA, Consumer-Duty, EP25/2, Equality-Act, LRTW, Owen, Côté, Charpentier, insurance-fairness-diag, motor, python]
description: "insurance-fairness-diag implements D_proxy (LRTW 2026, SSRN 4897265), Owen (2014) Shapley attribution, and Côté-Charpentier (2025) local vulnerability scores. It tells you how much proxy discrimination your pricing model contains and which rating factors are responsible — the diagnostic layer before you decide whether to correct it."
---

The FCA's EP25/2 (July 2025) evaluation of the general insurance pricing practices remedies made one thing clear: the regulator is no longer satisfied with high-level fair value assertions. The firms that handled the supervisory round well were the ones who could name the specific factors driving differential pricing outcomes across protected characteristic groups, quantify how much each factor contributed, and explain why the contribution was or was not actuarially justified.

Most UK pricing teams cannot do this. They can tell you whether their model uses a protected characteristic directly (it doesn't). They cannot tell you which of their twelve rating factors is leaking protected attribute information into the premium, or how much of the total discriminatory signal each factor accounts for.

[`insurance-fairness-diag`](https://github.com/burning-cost/insurance-fairness-diag) answers that question. It is the diagnostic layer for proxy discrimination: it measures how much discrimination your model contains and decomposes it to individual factors. It does not correct the discrimination — for that, see [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) or [`insurance-fairness-ot`](https://github.com/burning-cost/insurance-fairness-ot). But before you can correct a problem, you need to know its size and its source.

```bash
uv add insurance-fairness-diag
```

---

## The problem with current practice

Most proxy discrimination testing in UK personal lines pricing works like this: someone runs a correlation between a rating factor (say, annual mileage) and a protected characteristic (say, gender), finds a statistically significant correlation, and flags the factor as a risk. That is not proxy discrimination testing. That is correlation analysis. Correlation is not discrimination, and the two are not the same.

Proxy discrimination, in the Equality Act 2010 Section 19 sense, is about the pricing *outcome* — whether customers with a protected characteristic end up charged materially different premiums than otherwise-identical customers without that characteristic, and whether that difference is driven by the rating factors rather than by genuine risk differences.

The distinction matters because almost all insurance rating factors correlate with at least one protected characteristic. Postcode correlates with ethnicity. Annual mileage correlates with gender. Vehicle age correlates with socio-economic status. The question is not whether the correlation exists. The question is how much of the variation in your fitted premium is attributable to that correlation rather than to legitimate risk segmentation.

That is what D_proxy measures.

---

## D_proxy: the measure

Lindholm, Richman, Tsanakas and Wüthrich (SSRN 4897265, European Journal of Operational Research 2026) define D_proxy as the normalised L2-distance between your fitted price and the set of prices that could have been produced by an equivalent model with no access to the sensitive attribute.

Here is the intuition before the formula. Your fitted model h(x) assigns a premium to each policyholder. Now imagine a different model h*(x) that has the same overall structure but cannot distinguish between policyholders in different sensitive groups — it knows only what all policyholders in a group look like on average. The distance between h and h* measures how much of the variation in your actual premiums can be attributed to sensitive group membership rather than to legitimate factors.

Formally:

$$D_{\text{proxy}} = \frac{\sqrt{\mathbb{E}_w\left[(h^*(x) - \bar{h})^2\right]}}{\sqrt{\mathbb{E}_w\left[(h(x) - \bar{h})^2\right]}}$$

where h*(x) = E[h | S = s(x)] (the average predicted premium for all policyholders in the same sensitive group as x), and the expectations are exposure-weighted. This simplifies to the square root of the R-squared from regressing h on the sensitive attribute S: the fraction of premium variation that co-varies with S, on the square root scale.

D_proxy = 0 means your predicted premiums are completely independent of sensitive group membership. D_proxy = 1 means all the variation in your premiums is explained by group membership and none by within-group factor differences. In practice, values above 0.05 are worth investigating (AMBER threshold) and values above 0.15 warrant remediation consideration (RED threshold). The thresholds are not FCA-prescribed — they represent our judgement of materiality in a Consumer Duty context.

The library also gives you a 200-replicate bootstrap confidence interval, so you know whether a D_proxy of 0.07 is a statistically stable finding or artefact of estimation noise in a small portfolio.

---

## The Shapley attribution: the killer feature

D_proxy tells you the size of the discrimination problem. It does not tell you which factors are responsible. A D_proxy of 0.08 could be almost entirely driven by postcode area, or it could be distributed more or less evenly across all your rating factors. The remediation strategy is completely different in each case.

Owen (2014) — "Sobol' Indices and Shapley Value," SIAM/ASA Journal on Uncertainty Quantification 2(1) — provides the principled answer. His permutation estimator computes the contribution of each input variable to the total explained variance of a model. We use it here to decompose D_proxy across rating factors: the surrogate model learns to predict the between-group discriminatory component h* - μ_h from the rating factors, and then Owen Shapley effects attribute that discriminatory component to individual factors.

Each factor gets a phi value in [0, 1], normalised so they sum to 1 across all factors. A phi of 0.62 for postcode area means postcode is responsible for 62% of the total proxy discrimination in your model. That is the number you bring to a remediation discussion, a Consumer Duty fair value assessment, or an FCA supervisory conversation.

```python
import polars as pl
from insurance_fairness_diag import ProxyDiscriminationAudit

audit = ProxyDiscriminationAudit(
    model=my_glm,           # fitted sklearn-compatible model
    X=df,                   # Polars DataFrame with all columns
    y=df["claim_cost"],
    sensitive_col="postcode_area",   # the protected characteristic proxy
    rating_factors=["age_band", "vehicle_group", "ncd_years", "annual_mileage"],
    exposure_col="exposure",
)

result = audit.fit()
print(result.summary())
```

Output:

```
Proxy Discrimination Audit
  Sensitive attribute : postcode_area
  D_proxy             : 0.0821 (95% CI: [0.0714, 0.0931])
  D_proxy monetary    : £6.23
  RAG status          : AMBER

  Top discriminatory factors:
    1. annual_mileage: phi=0.4412 (£2.75) [red]
    2. vehicle_group: phi=0.2891 (£1.80) [amber]
    3. age_band: phi=0.1847 (£1.15) [amber]
    4. ncd_years: phi=0.0850 (£0.53) [green]
```

The D_proxy monetary figure translates the normalised scalar into premium units: at £6.23, an average policyholder's premium contains £6.23 worth of postcode-correlated pricing variation that cannot be attributed to within-group risk differences. The Shapley attribution then tells you that £2.75 of that comes from annual mileage, £1.80 from vehicle group, £1.15 from age band, and £0.53 from NCD years.

That is an actionable finding. Annual mileage is doing 44% of the discriminatory work. The remediation question — whether to adjust the mileage curve, apply a correction factor, or accept this as actuarially justified — is now informed by a specific number rather than a general anxiety about postcode correlations.

The phi RAG thresholds are: green below 0.10 (minor contributor), amber between 0.10 and 0.30, red at or above 0.30 (single-factor dominance). A single factor at red warrants specific justification in your Consumer Duty fair value assessment.

---

## How the attribution works

The Shapley computation runs on a surrogate model. We fit a RandomForestRegressor to predict D = h* - μ_h (the between-group discriminatory residual) from the rating factors. The Owen permutation estimator then samples random orderings of the rating factors, computes the marginal contribution of each factor when added to the coalition, and averages across permutations. With 256 permutations (the default), the estimator is accurate to within a few percentage points for up to around twenty factors.

The surrogate step is necessary because the original model may not decompose cleanly: a GBM or neural net does not expose the per-factor contributions we need in a form that integrates with the Owen estimator. The surrogate approach is standard practice for global sensitivity analysis of black-box models.

For tractability on large portfolios, the Shapley computation runs on a random subsample of 10,000 policyholders by default. This is configurable. On a 500,000-policy motor book, 10,000 is sufficient for stable phi estimates in our testing; the standard error on phi is typically below 0.02 at that sample size.

---

## Per-policyholder vulnerability scores

Beyond the portfolio-level D_proxy and the factor-level Shapley effects, the library also computes per-policyholder proxy vulnerability scores following Côté and Charpentier (2025).

The local score for each policyholder i is:

$$\phi_i = \frac{|h_i - h_i^*|}{h_i^*}$$

This measures how far each individual's actual premium deviates from the discrimination-free premium for their sensitive group, expressed as a fraction of that benchmark. A policyholder in a high-BME postcode area whose actual premium is 12% above the group average has a local vulnerability score of 0.12.

```python
# Per-policyholder scores
local = result.local_scores
print(local.select(["policy_id", "h", "h_star", "proxy_vulnerability", "rag"]).head(5))

# shape: (n_policyholders, 6)
# columns: policy_id, h, h_star, d_proxy_local, d_proxy_absolute, proxy_vulnerability, rag
```

These scores are useful for three purposes. First, they let you identify the tails of the distribution — the specific policyholders who are most affected, which matters for any targeted remediation. Second, they feed into the FCA's fair value assessment framework: Consumer Duty requires evidence about outcomes across customer groups, and the vulnerability score distribution by sensitive group is a direct measure of that. Third, they generate the audit trail that EP25/2 evidenced is absent from most firms' fair value processes.

---

## Unaware and aware benchmarks

The library also computes the unaware and aware premium benchmarks from Côté-Charpentier (2025).

The unaware premium is what a model trained *without* the sensitive attribute would predict. The aware premium is what a model trained *with* the sensitive attribute would predict. Your actual premium from an unaware model should sit between these benchmarks. If it is above the aware premium for a disadvantaged group, the model is discriminating in excess of even the explicitly trained benchmark — a strong signal for investigation.

```python
# Benchmark premiums
bm = result.benchmarks
print(f"Unaware mean: {bm.unaware_mean:.2f}")
print(f"Aware mean: {bm.aware_mean:.2f}")
```

---

## Generating the audit report

The `to_html()` and `to_json()` methods produce self-contained audit documents for FCA EP25/2 compliance purposes. The HTML report includes the D_proxy value with confidence interval, RAG status, factor-level Shapley attribution table, local score distribution, and the unaware/aware benchmark comparison. The JSON report contains the same data in machine-readable form for integration with `insurance-mrm`'s governance workflow.

```python
result.to_html("proxy_discrimination_audit.html")
result.to_json("proxy_discrimination_audit.json")
```

The JSON output includes the `sensitive_col`, `d_proxy`, `d_proxy_ci`, `rag`, and the full `shapley_effects` dictionary, keyed by factor name with `phi`, `phi_monetary`, `rank`, and `rag` for each. This is the evidence pack format the library was designed around: a reproducible, timestampable artefact you can store alongside the model version in your governance register.

---

## Where this fits in the fairness library stack

We now have three fairness-related libraries and it is worth being clear about what each one does.

**`insurance-fairness`** detects and corrects discrimination. It applies Lindholm (2022) marginalisation to produce discrimination-free prices, runs counterfactual tests, and generates FCA documentation. It answers: "what should we charge instead?"

**`insurance-fairness-ot`** corrects discrimination using optimal transport. The Wasserstein barycenter approach handles multi-attribute cases (adjusting for gender and disability simultaneously) and the causal path decomposition from Côté-Genest-Abdallah (2025) preserves actuarially justified effects. It answers: "what is the mathematically correct price correction?"

**`insurance-fairness-diag`** diagnoses discrimination without correcting it. It tells you how much you have (D_proxy), which factors are responsible (Shapley effects), who is most affected (local scores), and how you compare to naive benchmarks. It answers: "what exactly is going on in this model, and where?"

The diagnostic library is the right starting point. Before deciding whether to correct, you need to understand the magnitude and source of the problem. A D_proxy of 0.03 (green) on a model where one factor accounts for 80% of it is a different situation from a D_proxy of 0.11 (amber) spread evenly across eight factors. The remediation path, the proportionate justification analysis, and the Consumer Duty documentation requirements are all different.

Run the diagnostic first. Then decide whether the correction libraries are needed, and if so, which one is appropriate.

---

## Our view on the regulatory position

The FCA has not mandated a specific methodology for measuring proxy discrimination. EP25/2 identified the problem — firms' fair value frameworks lacked the granularity to evidence differential outcomes — without prescribing a technical fix.

D_proxy is the most theoretically grounded scalar measure available. It is derived from the LRTW framework that has become the standard academic reference for discrimination-free insurance pricing. It is interpretable: a normalised distance in [0, 1] with an intuitive decomposition. And it has a direct connection to the Equality Act Section 19 proportionate justification test — if you can show that the factors contributing high Shapley phi values are also the factors with the strongest actuarial justification, you have the structure of a defensible proportionate justification argument.

The Shapley attribution is the part most UK pricing teams do not have. Knowing D_proxy = 0.09 without knowing which factors drive it is like knowing your loss ratio is 82% without knowing which segments are unprofitable. The number without the attribution is not actionable.

---

## Installation

```bash
uv add insurance-fairness-diag
# or
pip install insurance-fairness-diag
```

Python 3.10+. Requires `polars >= 0.20`, `numpy >= 1.21`, `scikit-learn >= 1.3`. 1,911 lines of source, 137 tests. MIT licence. Source and notebooks at [github.com/burning-cost/insurance-fairness-diag](https://github.com/burning-cost/insurance-fairness-diag).

---

**See also:**
- [Discrimination-Free Pricing with Optimal Transport](/2026/03/10/insurance-fairness-ot/) — once you know which factors are leaking, this is how to correct the price
- [Your pricing model might be discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/) — the `insurance-fairness` library for direct Lindholm marginalisation and FCA documentation
- [Your model risk register is a spreadsheet](/2026/03/19/your-model-risk-register-is-a-spreadsheet/) — the JSON audit report from this library integrates with `insurance-mrm`'s governance workflow
