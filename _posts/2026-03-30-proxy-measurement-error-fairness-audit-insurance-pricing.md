---
layout: post
title: "Your Fairness Audit Is Underreporting Bias — Proxy Measurement Error in Insurance Pricing"
date: 2026-03-30
categories: [fairness]
tags: [fairness-audit, proxy-discrimination, measurement-error, LSOA, ethnicity, FCA, Equality-Act, insurance-fairness, sensitivity-analysis, arXiv-2603.17106, arXiv-2402.13391, indirect-discrimination, UK-insurance, python]
description: "Every UK fairness audit that uses LSOA ethnicity percentages as its protected attribute is reporting a lower bound on bias, not the true figure. Here is what that means in practice and what to do about it."
seo_title: "Proxy Measurement Error in Insurance Fairness Audits: Why Your Disparity Estimates Are Too Low"
---

Every UK fairness audit that uses LSOA ethnicity percentages as its protected attribute is reporting a lower bound on bias, not the true figure. The gap between that lower bound and the true disparity can be large. We think it is routinely large enough to matter for regulatory and commercial decisions.

This is not a theoretical concern. Xin, Mishler, Ritchie, Chouldechova, and Zhu (arXiv:2603.17106, March 2026) have shown formally — and demonstrated empirically on North Carolina voter registration data — that proxy-inferred protected attributes introduce systematic attenuation bias into disparity estimates. Their Proposition 1 gives the correction formula. The practical problem for UK insurers is that applying that correction requires validation data we do not have. But there is an approach that is actionable today, and it should become standard output for any UK fairness audit.

---

## What the attenuation actually looks like

Suppose your audit compares predicted premiums between two ethnic groups using postcode-level LSOA ethnicity percentages as a continuous proxy for individual ethnicity. You are not measuring individual group membership — you are measuring something correlated with it. When you compute a disparity ratio from noisy group labels, that noise suppresses the measured disparity towards zero.

Xin et al. derive the relationship in their Proposition 1. For the case where group membership is hard-classified from a proxy (e.g., classify a postcode as "majority South Asian" at 30% threshold), the observed disparity estimate relates to the true disparity through the proxy's confusion matrix C:

$$\tilde{\beta} \approx \text{diag}(C_n) \cdot C \cdot \text{diag}(C_n)^{-1} \cdot \beta_{\text{true}}$$

where C_n is the normalised confusion matrix and β̃ is what you observe. The mixing of groups through proxy misclassification compresses the spread of group-level estimates. Your audit reports that compressed figure.

More striking than the attenuation direction is what they find about BIFSG-imputed race in the North Carolina data: BIFSG **overestimates** the Black disparity by 29% relative to true self-reported race. The attenuation is not uniform and not always in the direction you might expect — the mixing artefact can inflate some group disparities while suppressing others, depending on the proxy method and underlying group distribution. The sign of the error is itself unknown until you have validation data.

---

## Why UK insurers cannot apply the correction directly

The Xin et al. correction formula requires a validation sample where you have *both* the proxy-inferred attribute and the true self-reported attribute for the same individuals. You then estimate C from that sample and apply the matrix inverse.

In North Carolina, the validation sample comes from mandatory racial self-identification on voter registration. There is no UK equivalent at individual level. Our closest analogues are:

- **GP patient records (NHS ethnicity codes)**: ~85% completeness nationally, but completeness varies substantially by region and practice. Not accessible to insurers for validation at individual policy level.
- **Insurance application data**: voluntary ethnicity disclosure rates are below 5% in most UK personal lines portfolios. Grossly insufficient for validation and subject to selection bias.
- **Financial Services Consumer Panel diversity surveys**: aggregate only. Cannot be linked to individual pricing outcomes.

The honest conclusion: UK insurers cannot apply the Xin et al. correction. We do not have the data. Pretending to apply it with assumed confusion matrix values would produce false precision. What we can do — and what we think should be mandatory practice — is acknowledge the uncertainty and quantify its bounds.

---

## The approach that is actionable today: sensitivity bounds

Patil and Iyengar (arXiv:2402.13391, "De-Biasing the Bias") take a different approach. Instead of requiring a known confusion matrix, they compute the *range* of possible true disparities across a grid of assumed proxy accuracy levels. You do not need to know whether your proxy has 70% or 80% accuracy. You report what the disparity would be under each assumption, and the reader can see how sensitive the conclusion is to that assumption.

This is a familiar technique in insurance modelling — we use sensitivity tables in reserving, in pricing risk margins, in stress testing. We should use them in fairness audits.

The practical implementation is straightforward. Take your point estimate of the disparity ratio and your best published estimates of proxy accuracy for your method. For ONOMAP (name-based ethnicity inference, common in UK financial services), published accuracy is roughly 70% at the broad ethnic group level. For LSOA-based geographic proxies, effective individual accuracy is harder to characterise but plausibly 55–75% depending on postcode heterogeneity. Use those as your bounds.

```python
import numpy as np

def proxy_disparity_bounds(
    observed_ratio: float,
    proxy_accuracy_low: float,
    proxy_accuracy_high: float,
    n_steps: int = 10,
) -> list[dict]:
    """
    Compute sensitivity bounds on a disparity ratio estimate across a grid of
    assumed proxy accuracy values.

    The observed_ratio is an attenuated estimate of the true disparity.
    Higher proxy accuracy → the observed ratio is closer to truth.
    Lower proxy accuracy → true disparity could be substantially larger.

    This is a first-order approximation: assumes symmetric attenuation and a
    two-group setting. For full multi-group bounds, see arXiv:2402.13391.
    """
    results = []
    for accuracy in np.linspace(proxy_accuracy_low, proxy_accuracy_high, n_steps):
        # attenuation factor: observed = accuracy * true + (1 - accuracy) * noise_term
        # under the simplest two-group model, observed_ratio ≈ accuracy * true_ratio + (1 - accuracy)
        # (the (1-accuracy) term reflects misclassified units pulling the ratio toward 1.0)
        if accuracy < 1.0:
            implied_true = (observed_ratio - (1 - accuracy)) / accuracy
        else:
            implied_true = observed_ratio
        results.append({
            "assumed_proxy_accuracy": round(accuracy, 3),
            "implied_true_disparity_ratio": round(implied_true, 4),
        })
    return results

# Example: audit reports disparity ratio of 1.08 using LSOA % South Asian proxy
bounds = proxy_disparity_bounds(
    observed_ratio=1.08,
    proxy_accuracy_low=0.55,
    proxy_accuracy_high=0.75,
)
for row in bounds:
    print(row)
```

A disparity ratio of 1.08 — which most governance committees would read as borderline and acceptable — implies a true ratio of 1.15–1.24 if the proxy accuracy is in the 55–75% range. That is no longer borderline. Whether it clears the substantive threshold for indirect discrimination under Equality Act 2010 section 19 is a legal question, but the committee is now making that judgement with the right numbers.

---

## What this means for the insurance-fairness library

Our `proxy_detection.py` functions — `proxy_r2_scores()`, `mutual_information_scores()`, `shap_proxy_scores()` — all accept a `protected_col` column and compute association metrics against it. None of them warn the user that if `protected_col` is itself a proxy estimate (e.g., LSOA % South Asian), all the scores they return are underestimates.

The problem is compounded: a rating factor with `proxy_r2 = 0.05` against LSOA% South Asian might have `proxy_r2 = 0.20` against true individual ethnicity. The audit is green-flagging factors that should be amber. This is not a minor numerical quirk — it is a structural bias in the audit output.

The same issue affects `IndirectDiscriminationAudit` in `indirect.py`. The class correctly requires a `protected_attr` column and assumes it is ground truth. If you pass a proxy-derived column as that argument, the proxy vulnerability scores it computes are lower bounds, not estimates. The class does not warn you of this.

Three changes are needed, in priority order:

**Immediate (documentation only)**: Add a `ProxyQualityWarning` that fires whenever the protected attribute column has a name or dtype that suggests it is a continuous proportion or proxy estimate rather than a verified categorical. Add a note to `IndirectDiscriminationAudit` docstring: *"If protected_attr is derived from proxy methods, treat all disparity estimates as lower bounds. Run sensitivity analysis to bound true disparity."*

**Short-term (code)**: Add a `proxy_disparity_bounds()` function to `proxy_detection.py` that takes a point estimate, a proxy accuracy range, and returns sensitivity intervals. The function above is a starting implementation.

**Medium-term (code)**: Add a `confusion_matrix` optional parameter to `disparate_impact_ratio()` and `demographic_parity_ratio()` in `audit.py`. When supplied, apply the Xin et al. Proposition 1 correction before computing ratios. When not supplied but proxy quality is declared below 1.0, apply sensitivity bounds instead.

The `proxy_r2_scores()` threshold calibration also needs revisiting. The current flag threshold of R² > 0.10 was set assuming `protected_col` is ground truth. Against a noisy LSOA proxy, that threshold should be lower — a factor that clears R² = 0.05 against a 65%-accurate geographic proxy might have R² ≈ 0.12 against true individual ethnicity, which should be flagged.

---

## What audit reports should say

A fairness audit section that uses proxy ethnicity data should contain language similar to this:

> Ethnicity is approximated using LSOA-level ethnic group proportions from Census 2021. This is a geographic proxy, not individual-level data. All disparity estimates reported below are lower bounds on true group-level disparity under the assumption that the proxy introduces classical attenuation bias. Using published estimates of LSOA-proxy accuracy (55–75% at individual level), the implied true disparity ratios range from [X] to [Y] against the reported point estimate of [Z]. These bounds are reported in the sensitivity table in Annex B.

This language is defensible under FCA Consumer Duty (PRIN 2A) and is consistent with the principle in Evaluation Paper EP25/2 that firms should be transparent about the limitations of their monitoring methods. It is also what an FCA reviewer or Equality Act litigant would expect to see.

What it is not is sufficient on its own. Reporting honest uncertainty is the floor, not the ceiling. The medium-term work is to build better data infrastructure: voluntary ethnicity disclosure with appropriate incentives, industry consortium approaches to linked-data validation, and engagement with the FCA on what a credible ground-truth programme looks like. But those are multi-year efforts. The sensitivity bound is something every team running a fairness audit can implement before the next governance cycle.

---

## The regulatory backdrop

FCA Evaluation Paper EP25/2 (February 2025) noted explicitly that firms' approaches to monitoring differential outcomes for groups with protected characteristics remain immature. Several firms reviewed were using proxy methods without any quantification of proxy quality or its effect on disparity estimates. The FCA has not yet mandated a specific approach, but the direction of travel is clear: qualitative acknowledgement of proxy limitations is no longer sufficient. Quantified uncertainty is expected.

The Equality Act 2010 section 19 indirect discrimination test applies to insurance pricing. A provision, criterion, or practice that puts persons sharing a protected characteristic at a particular disadvantage is unlawful unless it is a proportionate means of achieving a legitimate aim. That test is applied to actual group outcomes, not to proxy-estimated ones. A firm that demonstrates its pricing has no measurable disparity against a noisy proxy has not demonstrated compliance. It has demonstrated that it has not measured the problem properly.

We think the combination of sensitivity bounds (immediate), probabilistic proxy weights (medium-term), and industry-level linked-data validation (long-term) is the correct programme. The first of those costs an afternoon. There is no reason to wait.

---

*The sensitivity bounds approach discussed here draws on Patil and Iyengar (2024), "De-Biasing the Bias: On the Limits of Measuring Algorithmic Bias in Practice" (arXiv:2402.13391). The attenuation bias framework is from Xin, Mishler, Ritchie, Chouldechova, and Zhu (2026), arXiv:2603.17106. The `proxy_disparity_bounds()` function above is illustrative; a fuller implementation will land in the insurance-fairness library. The insurance-fairness library is on GitHub at [insurance-fairness](https://github.com/burning-cost/insurance-fairness).*
