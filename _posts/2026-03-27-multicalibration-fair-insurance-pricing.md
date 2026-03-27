---
layout: post
title: "Multicalibration: Portfolio Balance and Fairness Are the Same Test"
date: 2026-03-27
categories: [pricing, fairness]
tags: [multicalibration, autocalibration, fairness, proxy-discrimination, fca, consumer-duty, calibration, credibility, python]
description: "Denuit, Michaelides and Trufin (March 2026) unify autocalibration and non-discrimination into a single actuarial test. If your model fails it, you have a pricing problem and a regulatory problem at the same time."
---

There is a recurring failure mode in how UK pricing teams approach non-discrimination: they treat it as a compliance layer bolted on top of actuarial work. The statisticians build a model, then a separate team runs a proxy check, then someone attests to the regulator. The two activities share no methodology, no tooling, and often no language.

A paper published in March 2026 - Michel Denuit, Marie Michaelides and Julien Trufin, "Balance and Fairness through Multicalibration in Nonlife Insurance Pricing" ([arXiv:2603.16317](https://arxiv.org/abs/2603.16317)) - makes a stronger claim: that balance and fairness are the same requirement expressed at different levels of granularity. The concept is multicalibration, and it is worth understanding properly.

---

## Autocalibration: what pricing teams already know

A pricing model is autocalibrated if, when you group policies by their predicted premium, the average premium equals the average observed claims cost for every group. Formally:

```
E[claims | premium = p] = p   for all p
```

This is not controversial. It is what every UK pricing actuary means when they say the model is "on-level" or "in balance." You check it by plotting predicted versus observed across deciles of the predicted value - the A/E ratio by premium band should sit close to 1.0 throughout.

Autocalibration at the portfolio level is necessary but not sufficient. A model can have an aggregate A/E of 1.00 and be systematically over-charging one segment while under-charging another, as long as the two errors cancel at portfolio level. This is a calibration failure, regardless of whether the segments correspond to protected characteristics.

---

## Multicalibration: the same test, applied within every protected group

Multicalibration requires autocalibration to hold inside every protected group simultaneously:

```
E[claims | premium = p, group = g] = p   for all p, for all groups g
```

If young female drivers are in the £400-£500 premium band, the average claim among that group should be £400-£500. Not £350. Not £600. If it is not, the model is miscalibrated for that group at that premium level - and if the miscalibration is systematically in one direction, the group is being over- or under-priced.

The key insight from Denuit et al. is that this is not a fairness test imported from machine learning. It is autocalibration, the same concept pricing teams already rely on, applied with additional conditioning. A model that fails multicalibration has a pricing problem that would be considered a defect regardless of protected characteristics. The fact that the defect falls along protected-characteristic lines makes it discriminatory as well, but the actuarial objection is independent of that.

This matters for how you present findings internally. "Our model is miscalibrated for this group" lands differently from "our model may indirectly discriminate against this group." Both statements may be true simultaneously. The first one commands attention from pricing committees.

---

## What the test actually looks like

The implementation in Denuit et al. uses three approaches: local regression within groups, bias correction within groups, and credibility adjustments. The credibility approach is the most tractable for UK pricing teams because it maps directly to Bühlmann-Straub methods that actuaries already use.

The correction is:

```
corrected_premium = predicted * (z * AE_ratio + (1 - z) * 1.0)
z = min(n_observations / credibility_threshold, 1.0)
```

Small cells get a correction blended towards zero. Large cells get the full A/E adjustment. This prevents overfitting when a cell has, say, 45 policies - the AE ratio at that sample size is too noisy to act on confidently.

Our [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) library already has this implemented as `MulticalibrationAudit`. Here is what the audit workflow looks like:

```python
from insurance_fairness.multicalibration import MulticalibrationAudit
import numpy as np

audit = MulticalibrationAudit(
    n_bins=10,          # premium deciles
    alpha=0.05,         # significance threshold per cell
    min_bin_size=30,    # cells below this are flagged small_cell=True
    min_credible=1000,  # full credibility threshold for corrections
)

report = audit.audit(
    y_true=policies["incurred_claims"].values,
    y_pred=policies["pure_premium"].values,
    protected=policies["age_band"].values,  # or postcode_diversity, gender, etc.
    exposure=policies["earned_exposure"].values,
)

print(report.is_multicalibrated)       # True/False verdict
print(report.worst_cells)              # top 10 cells by |AE - 1|
print(report.group_calibration)        # per-group p-values
```

The output is a table with one row per (premium bin, group) cell, showing the AE ratio, n observations, p-value, and a `significant` flag. A cell is significant if the AE deviates from 1.0 at p < alpha and there are enough observations to be credible. If any cell is significant, the model is not multicalibrated.

To apply the correction:

```python
corrected = audit.correct(
    y_pred=policies["pure_premium"].values,
    protected=policies["age_band"].values,
    report=report,
    exposure=policies["earned_exposure"].values,
)
```

The correction modifies only the cells that failed. Everything else is left unchanged.

---

## The regulatory connection

The FCA's GIPP pricing attestation multi-firm review (December 2022) found that only 11 of the 66 firms reviewed produced records that met the FCA's expectations for attestation. The review covered 70% of the UK home and motor markets. The FCA's concern was with whether firms could demonstrate - not just assert - that their pricing complied with the rules.

The multicalibration test provides exactly the kind of structured, reproducible evidence that attestation requires. "We ran the multicalibration audit across 12 protected characteristics, 10 premium deciles, and 24 months of data. The model passes for 11 of 12 characteristics. Age band fails in the £400-£500 decile for 18-21 year olds, AE ratio 1.28 (n=1,847, p=0.003). We have applied a credibility correction reducing that cell's premium by 18%. Post-correction, all cells pass." That is an attestation-ready finding.

The test also provides a precise operationalisation of what proxy discrimination means. A model that uses postcode area, which is correlated with ethnicity, fails multicalibration if high-diversity postcodes have a systematically different AE ratio within a premium band than low-diversity postcodes. The discrimination manifests as a calibration failure - which is the form most likely to survive regulatory scrutiny, because it is grounded in actuarial methodology rather than imported fairness metrics.

Note that TR24/2 (August 2024), the FCA's product governance thematic review, also identified inadequate evidence frameworks at many firms. Multicalibration audits address both the attestation gap (GIPP) and the fair value evidence gap (Consumer Duty) with the same methodology.

---

## A note on the companion paper

Miao and Pesenti's simultaneous preprint "Discrimination-Insensitive Pricing" ([arXiv:2603.16720](https://arxiv.org/abs/2603.16720)) takes a more theoretical route. They frame fair pricing as an optimisation problem: find the pricing measure closest to the real-world measure (in KL-divergence) subject to the constraint that the pricing principle is insensitive to protected covariates. For multiple protected characteristics, they construct a barycenter of the discrimination-insensitive measures for each characteristic separately.

This is intellectually clean and the uniqueness result is useful, but the operational path is less clear. The KL-divergence barycenter framing requires a probability measure over claims, which in practice means fitting a full distributional model rather than just a point predictor. The Denuit et al. approach requires only a prediction and an observed outcome - which is what every pricing team already has.

We will cover the Miao-Pesenti framework in more detail when we look at extending the optimal transport module in [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness), where the transport correction machinery is already in place.

---

## What to do with your existing portfolio

The practical starting point is to run the audit on your renewal book, not your new business book. Renewal books have longer exposure periods, more stable compositions, and are the segment where pricing teams have most control over the trajectory.

Pick the protected characteristics that are material for your product. For UK motor, that is: age band (the most likely to produce a genuine failure), postcode diversity score (the proxy discrimination risk), and gender (legally prohibited as a direct rating factor since Test-Achats, but still worth checking as a residual).

Run the audit at 10 premium deciles. Look at cells with n > 100 first - small cells are noise. AE ratios outside 0.85-1.15 in a cell with 500+ policies are worth investigating. An AE of 1.28 on 1,847 policies (as in the example above) is not sampling variation.

If you find a significant cell, the correction is conservative by design. A credibility threshold of 1,000 means a cell needs 1,000 policies before you apply the full AE adjustment. With 1,847 policies and a full credibility threshold of 1,000, z = 1.0 and the correction is the full AE ratio. With 350 policies, z = 0.35 and you apply 35% of the adjustment.

The corrected premiums are not the final answer - they are the starting point for a pricing review. If the AE ratio is large in a specific cell, you need to understand why. Is the base GLM missing a rating factor interaction? Is there a data quality issue in that segment? Is the group genuinely higher risk than the model captures? The answer determines whether you correct the premium or fix the model. The multicalibration audit tells you where to look.

---

## The library

`MulticalibrationAudit` is in [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) and implements the Denuit-Michaelides-Trufin framework with exposure weighting, credibility-blended corrections, and polars DataFrames for the output tables. The code is tested on synthetic UK motor data with planted failures, which we will write up separately.

The audit produces a structured report - `is_multicalibrated`, `worst_cells`, `group_calibration` p-values, and the full bin-group table - that maps directly onto what you would put in an attestation or Consumer Duty evidence file.

---

**Reference:** Denuit, M., Michaelides, M. and Trufin, J. (2026). Balance and Fairness through Multicalibration in Nonlife Insurance Pricing. arXiv:2603.16317.

**See also:** Miao, W. and Pesenti, S. (2026). Discrimination-Insensitive Pricing. arXiv:2603.16720.
