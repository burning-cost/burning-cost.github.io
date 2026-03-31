---
layout: post
title: "Drift Localisation: Which Passengers Got Wet?"
date: 2026-03-31
categories: [conformal-prediction, model-monitoring]
tags: [conformal-prediction, drift-detection, model-monitoring, PSI, TRIPODD, insurance-monitoring, feature-attribution, sample-localisation, arXiv-2602-19790, ESANN-2026, python]
description: "Hinder et al. (arXiv:2602.19790, ESANN 2026) introduce bootstrap conformal p-values for identifying which individual observations are affected by drift. We explain why this is a different question from the one pricing teams actually ask — and when the distinction matters."
author: burning-cost
---

Your portfolio drift monitor fires. Something has shifted between Q2 and Q3. The question every pricing team asks at that point is: which features? Vehicle age? Postcode mix? Channel proportions? The question almost no pricing team asks is: which specific policies are the anomalous ones?

That asymmetry is not an accident. It reflects what pricing actuaries and model owners actually do with drift information. When vehicle age shifts, you reweight or retrain. When channel mix moves, you investigate your PCW bid strategy. In both cases, the actionable unit is the feature, not the individual record.

A paper from Hinder, Vaquet, Brinkrolf and Hammer — "Drift Localization using Conformal Predictions," arXiv:2602.19790, accepted at ESANN 2026 — addresses the other question: given a data stream with known drift, which samples are the drifting ones? Their answer uses conformal prediction in a way that is technically elegant. But before reaching for it, it is worth being precise about what problem it solves and whether that problem appears in your monitoring workflow.

---

## The two questions drift detection actually answers

Drift detection in insurance pricing collapses into two distinct questions that are often conflated under the same heading.

**Feature attribution** asks: which rating factors have shifted in their marginal distribution? Population stability index (PSI) answers this per feature, with a scalar summary and an amber/red threshold. The `InterpretableDriftDetector` in our `insurance-monitoring` library goes further, applying FDR-controlled hypothesis testing across all features simultaneously, weighted by exposure. TRIPODD (arXiv:2503.06606, implemented in `drift_attribution.py`) answers the related but sharper question: which features explain the change in model performance, not just input marginals?

**Sample localisation** asks: which individual observations are anomalous relative to the time period they belong to? This is a different question. The output is not a per-feature PSI score but a per-record p-value — a number for each policy saying how surprising it is, given the reference distribution, that this record appeared in the monitoring period.

Hinder et al. work on sample localisation. Their paper makes no claim to compete with PSI or TRIPODD. The confusion arises because both problems live under the "drift localisation" label in the literature, and the paper's framing — "which samples are drifting?" — sounds similar to "which features are drifting?" until you look at the output.

---

## How the method works

The conceptual move is clean. Define a sample as "drifting" if you can reliably identify which time period it came from using only its feature values. Formally, a sample $x$ belongs to the drift set $L$ if:

$$P(T \mid X = x) \neq P(T)$$

where $T$ is the time period label. If the time period is predictable from the features, the sample carries temporal information — it is distinguishable. If the time period is not predictable from the features, the sample looks the same in both periods — it is non-drifting.

This reduces drift localisation to binary classification: train a classifier to separate reference period from monitoring period observations. A sample confidently assigned to one period is a candidate drifter; a sample the classifier is uncertain about could plausibly come from either period.

The contribution of the paper is applying conformal prediction to this classifier. Rather than using the classifier's raw probability output as a heuristic anomaly score, they use it as the scoring function for a conformal test. This matters for one specific reason: conformal p-values have finite-sample validity guarantees. If you threshold at $\alpha = 0.05$, the false positive rate (flagging a genuinely non-drifting sample as drifting) is controlled at 5% under the exchangeability assumption. A standard isolation forest or LOF score gives you no such guarantee — you set a threshold by feel and accept the implied false positive rate without knowing it.

**The bootstrap calibration trick.** Standard split conformal wastes data: you partition observations into a training set and a calibration set, and the calibration set contributes to p-value computation but not model fitting. For the small samples typical in ablation benchmarks, this is costly. Hinder et al. instead use a bootstrap in-bag/out-of-bag structure:

For each of $n_\text{boot}$ bootstrap iterations:
1. Sample in-bag indices $I_\text{in}$ and out-of-bag indices $I_\text{oob}$
2. Train classifier $\hat{\theta}$ on $I_\text{in}$
3. For each in-bag sample $i$, accumulate a conformal p-value using $I_\text{oob}$ as the calibration set

The final p-value for sample $i$ is the median across bootstrap iterations. Because every sample is in-bag for some iterations and would be out-of-bag for others, the procedure makes full use of the data. Around 75–100 bootstrap iterations are sufficient for MLP classifiers before performance plateaus; decision trees require 300+.

The result is a per-sample reject/retain decision at controlled level $\alpha$, rather than an uncalibrated anomaly score.

---

## Why feature attribution is the primary workflow

In UK motor and property pricing, the features that move — and that matter when they move — are well understood: vehicle age distributions shift with supply chain disruptions and fleet renewal cycles; postcode mix moves with PCW algorithm changes and geographic expansion decisions; channel proportions respond to bid strategy; NCD distributions change with mid-term adjustment policy and new-to-market cohorts.

When these shift, the question "which features have changed, and by how much?" is directly actionable. PSI on the vehicle age histogram tells you the magnitude of the shift in interpretable units. TRIPODD decomposes that shift into its contribution to model performance degradation. Both outputs map directly onto decisions: retrain, reweight, flag for actuarial review, update trend indices.

The question "which specific policies are the anomalous ones?" is much harder to act on. Knowing that policies 47,823, 91,044 and 203,510 are anomalous does not tell you what to do with them unless you can identify a structural commonality. And if there is a structural commonality — they are all from the same aggregator, they are all commercial van policies mislabelled as personal lines — you will find it faster through feature attribution than through sample scoring.

This is why `insurance-monitoring` has PSI, CSI, KS tests, Wasserstein distances, TRIPODD and `InterpretableDriftDetector` — and has no per-sample drift scoring module. We built what pricing teams asked for.

---

## Where sample localisation genuinely adds value

There are two scenarios where the sample-level framing is the right tool.

**Contaminated data batches from aggregators.** Suppose a new comparison website partner begins injecting policies into your portfolio in week 8 of a quarter. The policies look superficially normal — they pass all standard validation checks — but they represent a meaningfully different risk population. The contamination is 4–5% of the monitoring-period volume.

PSI operates on marginal histograms across the full monitoring period. With 4–5% contamination spread across 10 histogram bins, the per-bin shift is under 0.5 percentage points. PSI will not fire. But a time-period classifier trained to separate the reference period from the monitoring period might pick up the contamination signal: if the contaminated records are sufficiently distinct from the reference distribution, they will be confidently classified as "monitoring period" even if the aggregate marginals look stable.

Sample-level CP then flags those records specifically, giving you a list of policy IDs for manual investigation. This is a data engineering / data quality use case rather than a pricing use case, but it is a real one.

**New distribution channel cohorts before PSI has power.** When a new channel comes on in month 1 of a quarter, you may have 500–2,000 records from it by the time you run your monthly monitoring cycle. PSI on 10-bin histograms needs reasonable mass in each bin to be reliable; at this volume, several bins will have under 20 records and the PSI statistic will be noisy. A conformal time-period classifier, trained on the reference period and the new cohort as the "monitoring period," can test whether the new cohort is distinguishable from the reference before you have enough data for PSI to be well-powered.

Both use cases require the CP classifier to be trained on a clean reference period — a data engineering assumption that is frequently violated. If the reference period itself contains contamination, the null distribution is wrong and the p-values are invalid.

---

## The Fish-Head problem

The paper's most instructive dataset is not Fashion-MNIST or the NINCO OOD benchmark. It is Fish-Head: a split of ImageNette's "tench" class by whether the fish's head faces left or right. This is subtle, semantically meaningful drift — the kind of drift that resembles real-world concept shift far more than clean domain-shift benchmarks.

Fish-Head requires substantially larger sample sizes than the other datasets before CP-localisation reliably exceeds random performance (n=250 per period, compared to n=60 for Fashion-MNIST and NINCO). The paper's honest acknowledgement of this is useful: subtle realistic drift is hard, and the method's power depends on the classifier's ability to separate the time periods in feature space.

In insurance, the analogous difficulty is high feature overlap between periods. Most renewing policies differ only marginally from their renewal-period predecessors: same vehicle, same postcode, similar NCD. The time-period classifier has to work hard to find the signal in a sea of nearly-identical observations. The method's power will be lower on insurance tabular data than on image embeddings — a point the paper does not address, because it evaluates only on images.

---

## What the existing monitoring stack covers

For completeness: the capabilities in `insurance-monitoring` that address drift and attribution questions in our current toolkit.

| Question | Tool |
|---|---|
| Has feature $k$ shifted marginal distribution? | PSI, CSI, KS test, Wasserstein |
| Which features explain model performance drift? | TRIPODD (`drift_attribution.py`) |
| Which features shifted with FDR control? | `InterpretableDriftDetector` (`interpretable_drift.py`) |
| Has model A/E drifted? | A/E ratios with Poisson CI (`calibration/`) |
| Has model discrimination drifted? | Gini drift z-test (`gini_drift.py`) |
| Are subgroup calibration gaps opening? | `MulticalibrationMonitor` |
| Which individual records are anomalous? | Not implemented |

The gap in the last row is genuine. We have chosen not to fill it because no pricing user has asked for it — and because the feature-attribution stack is comprehensive enough that sample-level scores would be redundant for all mainstream monitoring workflows. If a team comes to us with the specific problem of contamination detection — "we think an aggregator batch is polluting our Q4 data but PSI isn't moving" — then a sample-level CP module is worth building. We have estimated the implementation at around 200 lines of Python, no new dependencies beyond scikit-learn, and a few hours of work.

---

## The CP machinery in context

One detail worth noting for those using `insurance-conformal`: the bootstrap conformal calibration structure in this paper is reusable beyond drift localisation. The in-bag/out-of-bag bootstrap approach to producing per-sample CP scores — using each observation as both a training point (in-bag iterations) and a calibration point (out-of-bag iterations) — solves the data efficiency problem in split conformal for any low-sample-size application. If you need per-sample coverage scores and cannot afford to dedicate a held-out calibration set, this approach is worth understanding.

`insurance-conformal` currently uses split conformal throughout: `InsuranceConformalPredictor`, `TweedieConformPredictor`, and `ConditionalCoverageERT` all operate on a fixed calibration partition. For large insurance portfolios (50k–500k policies), this is fine — the efficiency loss from holding out 20% for calibration is negligible when you have 100,000 training observations. The bootstrap approach becomes relevant when you are operating on thin data: a new product with 800 policies, a specialist commercial line, a new postcode region with three years of history.

---

## The verdict

Hinder et al.'s method is technically sound. The CP framing is the right one for sample-level anomaly scoring: it produces valid p-values rather than heuristic thresholds, and the bootstrap calibration is a genuine contribution over naive split conformal for small samples. The paper is honest about its limitations — Fish-Head requires large n, image-only evaluation limits transferability claims, very sparse drifting-sample regimes remain hard.

For pricing model monitoring, the method answers a question that is secondary to the one teams actually ask. The primary question is feature attribution: which rating factors have shifted, by how much, and what does that mean for model performance? PSI, TRIPODD and `InterpretableDriftDetector` cover that ground well.

The secondary question — which specific records are anomalous? — matters for data quality audits and contamination detection. If that problem surfaces in a pricing team's workflow, we will build it. Until then, the paper is worth reading as a precise illustration of what sample-level and feature-level drift detection are, and why the distinction is not merely terminological.

The full paper is at [arXiv:2602.19790](https://arxiv.org/abs/2602.19790).
