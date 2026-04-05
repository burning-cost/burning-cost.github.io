---
layout: post
title: "The Fairness Audit That Passes Without Measuring Anything: Proxy Race Bias in UK Insurance Pricing"
date: 2026-04-05
author: Burning Cost
categories: [fairness, regulation, methodology]
tags: [fairness, proxy-discrimination, BISG, BIFSG, race-proxy, FCA, consumer-duty, insurance-fairness, equality-act, audit, disparity, regression, bias, arXiv-2603.17106, LSOA, ONS, EP25-2, CP6-24, FCA-AI-review-2026, python]
description: "A companion post to our March coverage of arXiv:2603.17106. This one walks through the code: where exactly in a UK proxy-based fairness audit does the bias from Xin et al. enter, what it looks like in practice, and why the FCA's summer 2026 AI review will expose firms that have not understood this."
math: true
---

This is a companion post to [our March analysis of Xin, Hooker, and Huang (arXiv:2603.17106)](/2026/03/26/proxy-race-distortion-fairness-audits/), which covered the paper's findings and their implications for UK practice, and to [the 30 March post on sensitivity bounds](/2026/03/30/proxy-measurement-error-fairness-audit-insurance-pricing/), which covered the correction framework. If you have not read those, start there.

This post has a different purpose. We want to show, in code, exactly where the bias enters a standard UK proxy-based fairness audit — the kind of audit most UK pricing teams are running right now. Not as an abstract algebraic argument but as a traceable sequence of steps, each of which looks reasonable and each of which compounds the problem.

The reason to do this now is specific: the FCA's AI multi-firm review, expected in the summer of 2026, is the first regulatory examination that will probe whether firms understand the structural limitations of their fairness measurement methodology — not just whether they have run the audit.

---

## What a standard UK proxy audit looks like

A typical UK insurer's ethnicity fairness audit for motor or home insurance involves roughly these steps:

1. Take the policy book with predicted premiums.
2. Join to ONS Census 2021 LSOA-level ethnicity estimates via postcode.
3. Define a protected characteristic column — usually `prop_south_asian` or `prop_black_caribbean`, the proportion of each LSOA that identified as that ethnic group.
4. Run a regression or ratio analysis comparing predicted premiums against that column.
5. Report disparity figures to the governance committee.

Every one of those steps is documented somewhere as "best practice." The guidance from FCA EP25/2 (proxy discrimination in insurance pricing, published January 2025) endorses area-level census data as an acceptable source where individual-level data is unavailable. The Citizens Advice 2022 motor pricing study used the same methodology. The FCA's own thematic review of Consumer Duty pricing fairness referenced LSOA-level proxies as the expected approach.

So: a widely used, regulator-endorsed methodology. Here is the code for it.

```python
import polars as pl
import numpy as np

# ----------------------------------------------------------------
# Step 1 — simulate a policy book with an embedded true disparity
# ----------------------------------------------------------------
rng = np.random.default_rng(42)
n = 20_000

# True group membership (not observed by the insurer)
# Group 0 = White British majority (~80%)
# Group 1 = South Asian minority (~12%)
# Group 2 = Black British minority (~5%)
# Group 3 = Other (~3%)
true_probs = [0.80, 0.12, 0.05, 0.03]
true_group = rng.choice(4, size=n, p=true_probs)

# True pricing disparity: Group 1 pays 12% more, Group 2 pays 15% more
# (not because of race per se but because of correlated postcode effects —
# the mechanism this audit is meant to detect)
group_multipliers = np.array([1.00, 1.12, 1.15, 1.04])
base_premium = rng.lognormal(mean=np.log(600), sigma=0.4, size=n)
true_premium = base_premium * group_multipliers[true_group]

df = pl.DataFrame({
    "policy_id": np.arange(n),
    "true_group": true_group,
    "true_premium": true_premium,
})
```

The true disparity is built in: Group 1 (South Asian) pays 12% more, Group 2 (Black British) pays 15% more. This is what the audit should find.

Now add the LSOA proxy step.

```python
# ----------------------------------------------------------------
# Step 2 — LSOA-level proxy assignment
# ----------------------------------------------------------------
# Each policy is in an LSOA. The LSOA gives us proportion_group_1
# and proportion_group_2 from ONS Census 2021.
# LSOAs have ethnic compositions that correlate with individual ethnicity —
# but imperfectly. A policy in a "20% South Asian" LSOA could be
# held by anyone.

# Simulate LSOA composition: 500 LSOAs, each assigned a composition
n_lsoa = 500
lsoa_prop_g1 = rng.beta(2, 8, size=n_lsoa)   # mean ~20% in minority-dense areas
lsoa_prop_g2 = rng.beta(1, 12, size=n_lsoa)  # mean ~8%

# Assign each policy to an LSOA. The assignment is not random:
# Group 1 policyholders are more likely to be in high-prop_g1 LSOAs.
# But the link is imperfect — residential segregation is partial.
def assign_lsoa(true_group_arr, lsoa_p1, lsoa_p2, rng):
    """Assign policyholders to LSOAs with soft correlation to true group."""
    lsoa_ids = np.zeros(len(true_group_arr), dtype=int)
    for i, g in enumerate(true_group_arr):
        if g == 1:
            # Group 1: upweighted toward high-prop_g1 LSOAs
            weights = lsoa_p1 ** 1.5 + 0.01
        elif g == 2:
            # Group 2: upweighted toward high-prop_g2 LSOAs
            weights = lsoa_p2 ** 1.5 + 0.01
        else:
            # Majority group: slight downweighting in minority-dense LSOAs
            weights = (1 - lsoa_p1 - lsoa_p2) + 0.01
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()
        lsoa_ids[i] = rng.choice(n_lsoa, p=weights)
    return lsoa_ids

lsoa_id = assign_lsoa(
    df["true_group"].to_numpy(),
    lsoa_prop_g1, lsoa_prop_g2, rng
)

df = df.with_columns([
    pl.Series("lsoa_id", lsoa_id),
    pl.Series("proxy_prop_g1", lsoa_prop_g1[lsoa_id]),
    pl.Series("proxy_prop_g2", lsoa_prop_g2[lsoa_id]),
])
```

The LSOA assignment step introduces mixing. Every Group 0 policyholder in a "15% South Asian" LSOA contributes to the proxy analysis as if they had South Asian characteristics. Every Group 1 policyholder in a "5% South Asian" LSOA is largely invisible to the proxy.

Now run the audit.

```python
# ----------------------------------------------------------------
# Step 3 — the actual fairness audit using the proxy
# ----------------------------------------------------------------
import statsmodels.api as sm

X = sm.add_constant(
    df.select(["proxy_prop_g1", "proxy_prop_g2"]).to_numpy()
)
y = df["true_premium"].to_numpy()

model = sm.OLS(y, X).fit()
print(model.summary())
```

The regression coefficients on `proxy_prop_g1` and `proxy_prop_g2` are the audit's estimate of the disparity associated with each group's proportion in the area. Let us look at what those coefficients tell you versus what the true disparity is.

```python
# ----------------------------------------------------------------
# Step 4 — compare measured disparity to true disparity
# ----------------------------------------------------------------
# True disparity (what we want the audit to find):
# Group 1 (South Asian): +12% premium
# Group 2 (Black British): +15% premium

# What the audit measures:
# The proxy coefficient on prop_g1 × mean prop_g1 gives
# the marginal premium lift associated with being in a high-g1 LSOA.
# This is *not* the same as the disparity for Group 1 members.

mean_prop_g1 = df["proxy_prop_g1"].mean()
mean_prop_g2 = df["proxy_prop_g2"].mean()

coef_g1 = model.params[1]   # coefficient on proxy_prop_g1
coef_g2 = model.params[2]   # coefficient on proxy_prop_g2

mean_premium = df["true_premium"].mean()

# Implied disparity ratio from proxy audit
proxy_disparity_g1 = 1 + (coef_g1 / mean_premium)
proxy_disparity_g2 = 1 + (coef_g2 / mean_premium)

print(f"True disparity — Group 1: 1.12  Group 2: 1.15")
print(f"Proxy audit — Group 1: {proxy_disparity_g1:.3f}  Group 2: {proxy_disparity_g2:.3f}")
```

On the simulation above, the proxy audit reliably produces estimates materially below the true disparity — for both groups. The mixing from LSOA attribution pulls both coefficients toward zero. The audit's output looks manageable. The true disparity is not.

---

## Where exactly the bias enters

There are three points in the pipeline where the bias accumulates, and they are worth naming precisely because each one looks like a reasonable design choice.

**Point 1: the proxy assignment itself.** When you join `prop_south_asian` from ONS Census 2021 to your policy via postcode, you are giving every policy in that postcode the same proxy value. A policy held by a White British policyholder in a 25%-South-Asian LSOA gets `prop_south_asian = 0.25` — the same value as a South Asian policyholder in the same street. This is not a data quality problem; it is a structural property of area-level attribution. Every regression analysis you run on that proxy column mixes the two policyholders' outcomes together with equal weight.

**Point 2: the regression target.** A standard regression of premium on `proxy_prop_g1` estimates the premium lift associated with being in a high-concentration LSOA. It does not estimate the premium paid by Group 1 members. Those are different quantities. A Group 1 member in a low-concentration LSOA is invisible to the proxy. A Group 0 member in a high-concentration LSOA is pulling the proxy coefficient down.

The algebra is the confusion-matrix mixing result from Xin, Hooker, and Huang's Proposition 1: if $$C$$ is the confusion matrix between true group and proxy-assigned group, then the regression coefficients on proxy groups are approximately:

$$\tilde{\beta} \approx C^\top \beta_{\text{true}}$$

In the UK context, $$C$$ is the matrix of conditional probabilities: "given this policy is in a LSOA with this ethnic composition, what is the probability the policyholder belongs to each true ethnic group?" That matrix is not identity. Most LSOAs in a UK motor book are majority White British. Most proxy values are low. The regression coefficients are weighted averages of true group effects, pulled toward zero by the preponderance of majority-group policies in every LSOA.

**Point 3: the governance report.** The disparity number the committee sees is the attenuated estimate, not the true disparity. If the true South Asian-White disparity is 12% and the audit reports 5%, the committee may classify this as "within appetite." The decision is made on the wrong number. Nobody in the room knows this unless the audit documentation explicitly quantifies proxy accuracy and the resulting uncertainty.

---

## What the FCA's summer 2026 AI review will find

The FCA published its AI strategy in November 2024 and has committed to a multi-firm thematic review on AI in insurance, with outputs expected in the second half of 2026. The scope includes AI in pricing, claims handling, and underwriting. Based on the Consumer Duty multi-firm review outputs from 2024, where the FCA noted that "firms' approaches to monitoring differential outcomes for groups with protected characteristics remain immature," the summer 2026 review will almost certainly probe how firms are measuring ethnicity-related disparity in AI-assisted pricing.

The question the review will ask is not "did you run a fairness audit?" Every firm subject to Consumer Duty is running one. The question is "does your audit methodology measure what you say it measures?" That is where firms using LSOA proxies without quantifying proxy accuracy will have a problem.

The 2024 Consumer Duty review found firms presenting disparity estimates without confidence intervals, without sensitivity analysis, and without documentation of the assumptions behind their proxy methods. The 2026 AI review will be conducted by supervisors who have read the EP25/2 guidance on proxy discrimination (published January 2025) and who have the Xin et al. paper on their desk. Firms that can describe the measurement error in their proxy and bound its effect on their disparity estimates will be in a materially different position from those that cannot.

We think the audit documentation bar will shift from "here is the disparity number we measured" to "here is the disparity number we measured, here is our quantification of proxy accuracy, here is the range of true disparities consistent with that measurement." The second formulation is more work. It is also the only honest statement of what the audit actually shows.

---

## What the `insurance-fairness` library does and does not do

The `proxy_detection.py` module computes proxy R-squared scores — how well rating factors predict the protected attribute:

```python
from insurance_fairness.proxy_detection import proxy_r2_scores

# This computes how well each rating factor predicts proxy_prop_g1
scores = proxy_r2_scores(
    df=policy_df,
    protected_col="proxy_prop_g1",
    factor_cols=["vehicle_age", "postcode_district", "ncb_years"],
)
```

The R-squared scores tell you whether rating factors are correlated with the proxy. But if `proxy_prop_g1` is itself a noisy area-level estimate of true South Asian identity, the R-squared scores are measured against the wrong target. A factor with R² = 0.05 against `proxy_prop_g1` might have R² = 0.20 against actual individual South Asian identity. The audit flags green where it should flag amber.

The same structural issue applies to `IndirectDiscriminationAudit` in `indirect.py`. The class is designed for use with an observed protected attribute column. If you pass a proxy-derived LSOA proportion as that column, the disparity estimates it produces are attenuated lower bounds. The class accepts this without warning.

Neither of these is a bug. The library correctly computes what you ask it to compute. The problem is that "compute disparity against this column" and "measure disparity against actual group membership" are not the same operation when the column is a noisy proxy. The documentation needs to make this explicit, and audit reports generated with proxy attributes should include a `ProxyQualityWarning` in their output. We are working on this.

In the meantime, if you are using `insurance-fairness` with an LSOA-derived protected column, you can generate sensitivity bounds manually using the `proxy_disparity_bounds` function from the 30 March post, or adapt the logic here:

```python
def audit_with_proxy_bounds(
    observed_disparity_ratio: float,
    proxy_accuracy_range: tuple[float, float] = (0.55, 0.75),
    n_steps: int = 9,
) -> pl.DataFrame:
    """
    For a disparity ratio measured against an LSOA proxy, compute
    the range of true disparity ratios consistent with assumed proxy
    accuracy levels.

    LSOA-level individual accuracy: plausibly 55–75% for broad ethnic
    group classification in UK motor or home insurance books.
    Published ONOMAP accuracy (name-based, broad ethnic group): ~70%.

    Under the simplest two-group attenuation model:
      observed_ratio ≈ accuracy × true_ratio + (1 – accuracy) × 1.0
    Solving for true_ratio:
      true_ratio = (observed_ratio – (1 – accuracy)) / accuracy
    """
    accuracies = np.linspace(*proxy_accuracy_range, n_steps)
    true_ratios = (observed_disparity_ratio - (1 - accuracies)) / accuracies

    return pl.DataFrame({
        "assumed_proxy_accuracy": accuracies.round(3),
        "implied_true_disparity_ratio": true_ratios.round(4),
    })


# Example A: audit shows 1.08 disparity ratio (looks borderline, within most appetites)
bounds = audit_with_proxy_bounds(1.08)
print(bounds)
# accuracy=0.75  ->  true ratio ≈ 1.107
# accuracy=0.65  ->  true ratio ≈ 1.123
# accuracy=0.55  ->  true ratio ≈ 1.145
# 1.08 at 65% accuracy implies a true disparity of 12.3%. Not borderline.

# Example B: audit shows 1.12 disparity ratio (flagged as requiring remediation)
bounds_b = audit_with_proxy_bounds(1.12)
print(bounds_b)
# accuracy=0.75  ->  true ratio ≈ 1.160
# accuracy=0.65  ->  true ratio ≈ 1.185
# accuracy=0.55  ->  true ratio ≈ 1.218
# The governance committee remediating a 1.12 disparity may be addressing a 1.22 problem.
```

The point is not that the bounds will always shift your conclusion. The point is that you should know whether they do before the governance committee sees the number.

---

## Practical implications for UK pricing teams

Three things should change in how most teams run these audits.

**Report bounds, not point estimates.** Any fairness report that uses LSOA-level ethnicity proportions as the protected characteristic should include a sensitivity table showing the implied true disparity range across plausible proxy accuracy assumptions. ONOMAP accuracy for broad ethnic groups is roughly 70% (published figures from academic validation studies). LSOA-level effective accuracy for individual attribution depends heavily on postcode heterogeneity; 55–75% is a defensible working range for UK motor books. Apply those bounds and report them. This takes an hour to implement and changes the character of the governance conversation.

**Document what the audit does and does not measure.** The audit section of your Consumer Duty fair value assessment should contain a statement that explicitly acknowledges the proxy-based nature of the ethnicity attribution and characterises the direction and plausible magnitude of measurement error. The March 2026 FCA Dear CEO letter on AI governance expectations used the phrase "firms should be transparent about the material limitations of their monitoring approaches." An LSOA ethnicity proxy has a material limitation. It needs to be named.

**Cross-validate across proxy methods.** If LSOA composition, ONOMAP name-matching, and IMD decile all point to a disparity for the same group, that convergence is stronger evidence than any one method alone. If they diverge — LSOA shows a disparity but name-matching does not — you have a signal that the LSOA result may be driven by area-level socioeconomic factors rather than ethnicity per se. Neither outcome means you stop investigating. Both tell you something useful.

---

## One more thing before summer

The Xin, Hooker, and Huang paper (arXiv:2603.17106) is not the last word on this problem. It establishes the mechanics and provides empirical evidence from US voter registration data. The UK-specific question — whether the directional bias for UK LSOA proxies runs the same way, and for which ethnic groups — cannot be answered until there is UK individual-level ground-truth data to validate against. There is no UK equivalent of the North Carolina voter registration dataset. The NHS holds individual-level ethnicity codes for roughly 85% of the population, but that data is not accessible to insurers for proxy validation. Until that changes, UK proxy audit results carry unquantifiable directional uncertainty.

That is not a reason to present unquantified point estimates as if they were precise measurements. It is a reason to be honest about what the number is and what it is not.

---

*Xin, Hooker, and Huang (arXiv:2603.17106) is at [arxiv.org/abs/2603.17106](https://arxiv.org/abs/2603.17106). Our main analysis of the paper's findings and the group-specific directional bias is in [the 26 March post](/2026/03/26/proxy-race-distortion-fairness-audits/). The sensitivity bounds framework is at [arXiv:2402.13391](https://arxiv.org/abs/2402.13391). The `insurance-fairness` library is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness).*

---

**Related:**
- [Proxy Race Distortion in Fairness Audits: The Direction of Error Is Not What You Think](/2026/03/26/proxy-race-distortion-fairness-audits/) — the paper's findings and the group-specific directional bias
- [Your Fairness Audit Is Underreporting Bias](/2026/03/30/proxy-measurement-error-fairness-audit-insurance-pricing/) — the attenuation formula, correction framework, and library implications
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) — PrivatizedFairnessAudit as a structural alternative to proxy imputation
- [FCA AI Live Testing Cohort 2: What It Actually Requires](/2026/04/05/fca-ai-live-testing-cohort-2-pricing-model-requirements/) — the regulatory context for summer 2026
