---
layout: post
title: "Proxy Race Distortion in Fairness Audits: The Direction of Error Is Not What You Think"
date: 2026-03-26
categories: [fairness, regulation, methodology]
tags: [fairness, proxy-discrimination, BISG, BIFSG, race-proxy, FCA, Consumer-Duty, insurance-fairness, Equality-Act, audit, disparity, regression, bias, arxiv]
description: "A new paper (Xin, Hooker, Huang 2026) shows that BIFSG proxy race distorts regression-based fairness audits in two distinct mechanisms  -  and the direction of distortion is group-specific and unpredictable. The common assumption that proxy audits uniformly attenuate disparity is wrong for Black groups."
---

A paper from Xin, Hooker, and Huang (arXiv:2603.17106, March 2026) should change how actuarial teams interpret the output of any regression-based fairness audit that uses proxy-imputed race. The finding is not subtle: proxy imputation via BISG or BIFSG introduces systematic distortion into disparity estimates, and the direction of that distortion is group-specific. For some groups it attenuates apparent disparity; for others it amplifies it. You cannot tell which without knowing the structure of the misclassification errors for your specific proxy and your specific book.

This matters for UK pricing teams even though BISG and BIFSG are American constructs. The mechanism that causes the distortion is algebraic, not empirical. It operates on any proxy with less than perfect accuracy. UK firms using Census 2021 LSOA-level ethnicity composition as a continuous proxy, or ONS name-list surname matching, are not immune.

---

## Two mechanisms, one paper

The paper identifies two distinct channels through which proxy-imputed race corrupts a regression-based fairness audit.

**Mechanism 1  -  intrinsic mixing.** BIFSG assigns each individual a predicted group based on surname, first name, and census geography. When someone from true group k is assigned to predicted group j, their outcome enters j's regression. The proxy group's coefficient is therefore a confusion-matrix-weighted average of the true group coefficients. Because White is the largest group in the US data (1.85 million NC voter records), it dominates every weighted average. Every minority group's proxy coefficient is pulled toward the White baseline. This is not a data artefact  -  it is arithmetic. It happens on any book where one group is numerically dominant.

**Mechanism 2  -  structured correlated errors.** BIFSG's misclassification rate is not random. It varies systematically with ZIP code racial composition and socioeconomic status. Errors are not mean-zero conditional on outcomes. This generates spurious associations in the regression output that go beyond what mixing alone predicts. The pattern of errors reflects the joint distribution of surnames, geography, and socioeconomic variables in the underlying Census data.

These two mechanisms are separable analytically, and only the first transfers directly to UK practice. Mechanism 2 is specific to the BIFSG architecture and to US Census geography. The UK equivalents  -  LSOA-level ethnic composition from Census 2021, or ONS name lists  -  have different error structures, and whether their errors are mean-zero conditional on outcomes is an open empirical question that has not been studied with UK insurance data.

---

## The finding that matters most

The headline most commentators took from early versions of this work  -  "proxy audits underestimate disparity by 20–40%"  -  is wrong as a general statement. The paper's actual finding is more uncomfortable.

Using NC voter data with ground-truth race (1.85 million individuals), the authors compare audit results under true race against audit results under BIFSG-imputed race, with SES controls. The results:

- **Black disparity is overestimated by approximately 31%** under proxy. The proxy audit overstates the Black-White gap.
- **Asian and Hispanic disparities are underestimated** under proxy. The proxy audit understates those gaps.
- **BIFSG accuracy varies by group**: White 86%, Black 74%, Hispanic 74%, Asian 67%, Others 20%.

There is no uniform direction. An audit using BIFSG-imputed race may simultaneously overstate disparity for one group and understate it for another. A firm that concludes "our proxy audit shows no material disparity for Asian or Hispanic customers" may be presenting a result that is directionally wrong. A firm that concludes "we found a modest Black-White disparity and have addressed it" may have overreacted to a statistical artefact.

This is the paper's most important practical implication. The sign of the bias is not something you can assume; it depends on the confusion matrix structure for your specific proxy, which depends on the name and geography distribution of your book.

---

## What transfers to UK practice

The UK regulatory frame is Equality Act 2010 Section 19 indirect discrimination plus FCA Consumer Duty (PRIN 2A), not US ECOA or HMDA. UK insurers do not use BISG or BIFSG. But they do use proxies: area-level ethnic composition from ONS Census 2021, postcode-level IMD scores as socioeconomic proxies, and in some cases surname-matching against ONS name lists for South Asian and Chinese name frequencies.

**Mechanism 1 transfers directly.** Any proxy with less than perfect individual-level accuracy will generate confusion-matrix-weighted mixing of true group effects into proxy group coefficients. The mathematics does not care whether the proxy is US surname-geography or UK LSOA composition. If the proxy misclassifies 30% of individuals (plausible at the individual level for any area-level attribution), the regression coefficients for minority groups are pulled toward the majority. For UK motor and home insurance, where White British is the dominant group by policy count, the pull is toward White British.

**Mechanism 2 does not transfer directly.** The UK equivalents have different error structures. LSOA-level ethnic composition gives a continuous proportion rather than a probabilistic group assignment; the misclassification pattern is different in kind. This is not to say UK proxies are unbiased  -  only that the specific structured error pattern documented in the paper for BIFSG does not apply to LSOA attribution. No equivalent UK study exists. There is no UK dataset that permits the NC-style ground-truth validation: unlike the US voter registration data, there is no individual-level ethnicity registry that can be used to validate proxy accuracy at scale.

The implication is that UK proxy audit results carry uncertainty that cannot currently be quantified. You know that mixing distortion operates in principle. You do not know whether UK LSOA attribution overestimates or underestimates specific ethnic group disparities in your book, because the validation data does not exist.

---

## What to do about it

A related paper  -  De-Biasing the Bias (Belloum, Louppe, and Frénay, arXiv:2402.13391, 2024)  -  proposes a practical approach for exactly this situation: sensitivity analysis bounds rather than point estimates.

The idea is to treat the proxy accuracy parameters as partially unknown, construct a range of plausible confusion matrices based on what you do know (proxy accuracy by group, where you have it; Census ethnic group proportions), and report disparity estimates as intervals rather than point estimates. A disparity of 8% that could plausibly be anywhere between 3% and 18% under realistic proxy error scenarios is a different regulatory communication than an apparent 8% presented as a precise figure.

For UK pricing teams, the practical implication is this: treat proxy-regression disparity estimates as diagnostic screens with bounds, not as point estimates. If a proxy audit shows no material disparity, that is evidence but not proof  -  because the direction and magnitude of proxy-induced distortion is unknown for your book. If a proxy audit shows a material disparity, the first question to investigate is whether the result is consistent across multiple proxy methods (LSOA composition, name matching, IMD decile) before treating it as a reliable signal requiring remediation.

The FCA's EP25/2 does not specify methodology at this level of detail, but the expectation  -  documented in TR24/2 (August 2024)  -  is that firms take reasonable steps to understand the limitations of their methodology. "We used a proxy and report the number" is not taking reasonable steps. "We used a proxy, we quantified the uncertainty in that proxy's accuracy, and we present our results with explicit bounds" is.

---

## What this means for our library

The `IndirectDiscriminationAudit` in `insurance-fairness` requires an actual protected attribute column  -  it does not impute. This is the architecturally correct choice and the paper reinforces it: if you feed `IndirectDiscriminationAudit` an ONS LSOA ethnicity proportion, you are feeding it a proxy, and the audit output inherits the distortions the paper documents. The library will not prevent this; it accepts any numeric column as the protected characteristic.

The thing that currently needs strengthening is the proxy quality reporting when the protected attribute column is itself an imputation. `proxy_detection.py` reports R-squared scores for how well rating factors predict the protected column  -  but when the protected column is itself a LSOA-level estimate, those R-squared scores are measuring association with an already-noisy target. They are systematically underestimates of the true association with underlying ethnicity. The reported R-squared for `postcode_district` against `ethnicity_prop` may be 0.38 (as in the example in our EP25/2 guide), but if `ethnicity_prop` is a noisy individual-level approximation of true ethnicity, the true association could be materially higher.

We intend to add a `proxy_quality_warning` field to the `FairnessReport` that surfaces this issue when the protected column shows evidence of being itself a coarse proxy: low within-LSOA variance, integer-valued proportions, or other characteristics that indicate area-level rather than individual-level attribution. In the meantime, the correct practice is to document explicitly when the protected attribute is area-attributed rather than individually observed, and to treat the disparity estimates accordingly.

---

## Our view

The paper is a significant methodological contribution. Its most useful finding is the one that contradicts the intuition: proxy audits do not uniformly attenuate disparity estimates. They can amplify disparity for some groups while attenuating it for others, depending on the confusion matrix. For Black policyholders in the NC sample, the proxy audit overstated disparity by 31%. For Asian and Hispanic policyholders, it understated disparity.

That asymmetry is uncomfortable because it means the safe assumption  -  "our proxy audit is probably conservative; the true disparity is probably no worse than what we measured"  -  is not valid in general. Whether it is valid for your book depends on which groups are over- and under-classified by your specific proxy, which depends on the name and geography distribution of your specific portfolio.

Until UK-specific validation data exists (and we do not expect it to appear soon  -  there is no equivalent UK voter registration dataset), pricing teams should report proxy audit results with explicit acknowledgement of directional uncertainty, and should cross-validate disparity estimates across multiple proxy methods where possible. A disparity that appears across LSOA composition, IMD decile, and name-matching proxies simultaneously is more credible than one that appears in only one of them.

This is not a reason to abandon proxy audits. It is a reason to understand what they are: noisy estimates with model-dependent bias, not ground-truth measurements.

---

Xin, Hooker, and Huang (arXiv:2603.17106) is available at [arxiv.org/abs/2603.17106](https://arxiv.org/abs/2603.17106). The De-Biasing the Bias sensitivity framework is at [arxiv.org/abs/2402.13391](https://arxiv.org/abs/2402.13391).

The `insurance-fairness` library is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness).

---

**Related posts:**
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/)  -  the PrivatizedFairnessAudit approach using local differential privacy, which avoids proxy imputation entirely
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/)  -  the `insurance-fairness` library: LRTW framework, proxy R² detection, and discrimination-free pricing
- [FCA Proxy Discrimination Testing in Python: A Practical Guide](/2026/03/22/fca-proxy-discrimination-python-testing-guide/)  -  EP25/2 conformant audit workflow with `FairnessAudit`
- [Three Ways to Detect Proxy Discrimination: When They Agree and When They Don't](/2026/03/22/three-methods-proxy-discrimination-detection-compared/)  -  mutual information, proxy R-squared, and SHAP proxy scores: what they each measure and when they diverge
