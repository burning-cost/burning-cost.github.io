---
layout: post
title: "SIU Referral Lists With a Provable False Discovery Rate: Conformal P-Values for Claims Fraud"
date: 2026-03-11
categories: [libraries, claims, fraud]
tags: [conformal-prediction, fraud-detection, SIU, FDR, BH, anomaly-detection, isolation-forest, mondrian, IFB, consumer-duty, FCA, PRIN2A, insurance-conformal-fraud, python, motor, claims]
description: "Every SIU referral threshold in UK motor insurance is arbitrary — 'refer the top 5%' or 'anything above 0.7'. Conformal p-values fix this by giving each claim a statistically valid p-value under the null hypothesis of genuine claim. Apply Benjamini-Hochberg and you get a referral list with a provable false discovery rate guarantee. insurance-conformal-fraud is the first Python library to do this for insurance, including Mondrian stratification by claim type, integrative p-values from SIU case files, and Fisher combination for IFB consortium detection."
---

Every SIU manager in UK motor insurance has, at some point, been asked to justify a referral threshold. The answer is almost always some variant of: "We refer the top 5% by fraud score" or "anything above 0.7 from the model." If pushed, the answer becomes "that's what the team can handle" or "it's what we've always done."

What it is not is a statistical guarantee on the false referral rate.

That matters for two reasons. First, PRIN 2A.9 of the FCA's Consumer Duty requires firms to demonstrate that their claims handling produces good customer outcomes. Referring a genuine customer to SIU is foreseeable harm. An insurer who cannot quantify the rate at which it happens has a weak compliance position. Second, operationally, an arbitrary threshold is either too tight (missing fraud) or too loose (wasting SIU capacity on genuine claims). Without a principled calibration method, you are guessing.

Conformal p-values are the fix. [`insurance-conformal-fraud`](https://github.com/burning-cost/insurance-conformal-fraud) applies them to claims fraud detection with formal false discovery rate (FDR) guarantees.

```bash
pip install insurance-conformal-fraud
```

---

## What conformal p-values actually give you

Start with a calibration set of confirmed-genuine claims, claims that have been investigated and cleared. Train any sklearn-compatible anomaly detector, IsolationForest is the standard starting point, on a separate training set. Then, for each new claim, compute the fraction of calibration claims that were more anomalous than it:

    p_i = ( #{j in C : s_j >= s_i} + U_i ) / (|C| + 1)

where `s_j` is the nonconformity score (e.g., the negated Isolation Forest depth), `C` is the calibration set, and `U_i` is a uniform tie-breaking draw.

This p-value is marginally valid: under the null hypothesis that claim `i` is genuine, `P(p_i <= alpha) <= alpha` for all alpha. This is not an asymptotic result, not a large-sample approximation. It holds in finite samples given exchangeability between calibration and test genuine claims. The result is from Bates, Candes, Lei, Romano and Sesia, Annals of Statistics 51(1):149-178, 2023.

The practical meaning: a claim with `p_i = 0.02` is more anomalous than 98% of confirmed-genuine calibration claims. Under the null, you would see a p-value this small 2% of the time by chance alone.

Now apply the Benjamini-Hochberg procedure across all test claims. Sort p-values in ascending order. Find the largest rank `k` such that `p_(k) <= k * q / n`. Flag claims 1 through `k` for SIU referral. This controls FDR at level `q`: among all flagged claims, at most a fraction `q` are expected to be genuine. That is a provable guarantee, not an empirical calibration on historical data.

The `ConformalFraudScorer` in `insurance-conformal-fraud` handles the full workflow:

```python
from sklearn.ensemble import IsolationForest
from insurance_conformal_fraud import ConformalFraudScorer, bh_referrals

# Separate fit and calibrate steps — keep them clean
scorer = ConformalFraudScorer(IsolationForest(n_estimators=200, random_state=42))
scorer.fit(X_train_genuine)
scorer.calibrate(X_cal_genuine)

# Per-claim conformal p-values
pvalues = scorer.predict(X_new)

# BH referral list at 5% FDR
referrals = bh_referrals(pvalues, fdr_target=0.05)
print(f"Referring {referrals.sum()} claims. Expected false referrals: {0.05 * referrals.sum():.1f}")
```

The Storey-BH variant estimates the null proportion `pi_0` from the p-value distribution to get tighter (less conservative) rejections:

```python
from insurance_conformal_fraud import storey_bh_referrals
referrals = storey_bh_referrals(pvalues, fdr_target=0.05)
```

---

## Why stratification by claim type is not optional

Running a single IsolationForest across TPBI, accidental damage, and theft claims is, statistically speaking, a mistake.

TPBI (third-party bodily injury) fraud is primarily organised: crash-for-cash, exaggerated whiplash, induced accidents. The relevant features are medical disbursements, hire car duration, solicitor involvement, and claimant count. The base rate is around 1-3% in UK motor books.

Accidental damage fraud is primarily opportunistic: owner give-ups, inflated repair quotes, fictitious damage. Relevant features are repair cost relative to vehicle value, garage type, and vehicle age.

Theft fraud has a different signature again: fictitious thefts and owner give-ups with high vehicle values, featuring patterns around police report timing, GPS data, and settlement speed.

These claim types have different base rates, different feature distributions, and different fraud patterns. A calibration set mixed across all three will not be exchangeable with test claims from any single type. The conformal p-value validity guarantee depends on exchangeability. Mixing strata violates it.

The `MondrianFraudScorer` enforces separate calibration per stratum and combines the p-values before applying BH:

```python
from insurance_conformal_fraud import MondrianFraudScorer, bh_referrals

scorer = MondrianFraudScorer(
    base_scorer=IsolationForest(n_estimators=200, random_state=42),
    strata=["TPBI", "AD", "Theft"]
)
scorer.fit(X_train, strata=train_strata)
scorer.calibrate(X_cal, strata=cal_strata)

pvalues = scorer.predict(X_new, strata=new_strata)
referrals = bh_referrals(pvalues, fdr_target=0.05)
```

The minimum viable calibration set is around 100 confirmed-genuine claims per stratum. Below that, the p-values are still valid but the intervals around the FDR estimate widen considerably. If your theft book is small, you may need to aggregate across years or use an industry calibration set, accepting that you are conditioning on a less current fraud signature.

---

## Using SIU case files to sharpen detection power

The basic conformal approach treats fraud detection as one-class classification: learn the distribution of genuine claims and flag outliers. This ignores the most valuable information any SIU team has: confirmed fraud cases from prior years.

If your book has five years of SIU case files, those represent hundreds or thousands of confirmed fraud examples with known fraud patterns. Ignoring them because the core algorithm does not accommodate labelled positives is waste.

The `IntegrativeConformalScorer` implements the method from Lemos et al., JRSS Series B 86(3):671, 2024. It adaptively combines a one-class scorer (trained only on genuine claims) with a binary classifier (trained on genuine plus confirmed fraud), selecting weights based on which provides more discriminative power for each test claim:

```python
from insurance_conformal_fraud import IntegrativeConformalScorer

scorer = IntegrativeConformalScorer()
scorer.fit(
    X_genuine=X_train_genuine,
    X_fraud=X_siu_confirmed  # confirmed SIU case files
)
scorer.calibrate(X_cal_genuine)

pvalues = scorer.predict(X_new)
referrals = bh_referrals(pvalues, fdr_target=0.05)
```

The FDR guarantee holds regardless of which classifier dominates — the adaptive weighting is part of the valid procedure. What you gain is statistical power: more genuine fraud cases detected at the same FDR target. In practice, with a well-populated SIU case file, this tends to materially reduce the p-values on organised fraud rings because the binary classifier has seen their patterns directly.

One practical constraint: the SIU case files must represent the kinds of fraud you expect to see in the current book. Historical case files from the pre-OIC portal period (before the May 2021 reforms to the whiplash injury claims process) have a different TPBI fraud signature to the post-reform period. Using 2018-2020 TPBI fraud cases to calibrate a 2025 model will help less than you might hope and may hurt if the fraud patterns have inverted.

---

## Combining signals across the IFB consortium

Fraud rings operate across multiple insurers simultaneously. A staged accident involves claimants who have policies at Admiral, Direct Line, and Aviva. Each insurer sees partial signal. Admiral may have a slightly anomalous TPBI claim. Direct Line has another. Aviva has a third. Individually, none triggers SIU referral. Combined, the pattern is obvious.

The Insurance Fraud Bureau runs the Claims and Underwriting Exchange (CUE), which provides a matching key: VRN plus accident date plus postcode typically identifies the same incident across insurers. The operational infrastructure to share signals exists. What has not existed is a statistically valid method to combine signals without sharing raw claims data.

Fisher's method on p-values provides it. If insurer 1 computes `p_1` and insurer 2 computes `p_2` for the same claim, independently of each other, then under the null:

    T = -2 * (log(p_1) + log(p_2)) ~ chi-squared(4)

More generally, for `n` insurers, the test statistic is chi-squared with `2n` degrees of freedom. The combined p-value is `P(chi-squared(2n) >= T)`. Apply BH to the combined p-values across all shared claims.

What each insurer shares is one number per claim per insurer: a conformal p-value. No raw feature data. No model architecture. No claim details. GDPR-compliant and operationally straightforward.

```python
from insurance_conformal_fraud import FisherCombiner, bh_referrals

# Each insurer runs their own scorer, shares only p-values
p_aviva = aviva_scorer.predict(X_shared_claims)
p_admiral = admiral_scorer.predict(X_shared_claims)
p_directline = directline_scorer.predict(X_shared_claims)

combiner = FisherCombiner()
combined_pvalues = combiner.combine([p_aviva, p_admiral, p_directline])
referrals = bh_referrals(combined_pvalues, fdr_target=0.05)
```

The independence requirement across insurers is satisfied when each insurer calibrates on their own confirmed-genuine claim set, with no overlap. Within-insurer p-values remain positively correlated (as handled by the Bates et al. result), but cross-insurer p-values for the same claim are independent of each other given separate calibration data.

---

## The Consumer Duty compliance statement

The `FraudReferralReport` generates structured output for compliance teams:

```python
from insurance_conformal_fraud import FraudReferralReport

report = FraudReferralReport(
    pvalues=pvalues,
    referrals=referrals,
    fdr_target=0.05,
    claim_ids=claim_ids,
    model_description="IsolationForest(n_estimators=200), Mondrian by claim type"
)

report.to_html("siu_referrals_march_2026.html")
```

The HTML output includes: the FDR guarantee statement ("among the N claims referred, we expect at most 5% to be genuine under the stated exchangeability assumption"); the distribution of p-values; which stratum each referral came from; and an explicit statement of the assumptions on which the guarantee depends.

That last part matters. The Consumer Duty compliance statement is honest about what the guarantee is not. It is not a guarantee that all high-p-value claims are genuine. It is not immune to calibration drift. It is a finite-sample, model-free guarantee conditional on the exchangeability of calibration and test genuine claims. If that assumption fails, the guarantee fails too.

---

## The assumption you need to take seriously

Exchangeability between the calibration set and test genuine claims is the load-bearing assumption. The p-value validity guarantee is exactly as strong as this assumption and no stronger.

In UK motor insurance, this assumption fails in predictable ways. The most significant: fraud patterns evolve, and they evolve faster after regulatory changes. The May 2021 OIC portal reforms capped whiplash compensation at amounts most claims farmers no longer consider worth pursuing for straightforward soft tissue injury. This shifted organised fraud toward more complex injuries and different claim types. A calibration set from 2019-2020 has a different genuine-claim distribution to 2025 in ways that matter for the nonconformity scores.

The library includes calibration drift monitoring via power martingale:

```python
from insurance_conformal_fraud import CalibrationMonitor

monitor = CalibrationMonitor()
for p in rolling_genuine_claim_pvalues:
    alert = monitor.update(p)
    if alert:
        print("Exchangeability suspect — consider recalibration")
```

The martingale grows when p-values deviate from uniformity under the null. Vovk's suggested alert threshold is 20. If you hit it, your calibration set is probably stale.

The other failure mode is contaminated calibration: confirmed-genuine includes undetected fraud. If 10% of your calibration claims are actually fraud that was never caught, your nonconformity scores for genuine claims are inflated, your p-values for genuine test claims are deflated, and your FDR guarantee is compromised. The library documentation is explicit about this. "Confirmed genuine" means investigated and cleared, not uninvestigated.

---

## Practical requirements

The library wraps `nonconform` for the core p-value machinery. Full credit to Oliver Hennhoefer's work there — `insurance-conformal-fraud` adds the insurance domain layer rather than reimplementing conformal p-value mechanics from scratch.

Minimum calibration set size: 200 confirmed-genuine claims for the basic scorer. Per stratum for Mondrian: 100. The IntegrativeConformalScorer benefits from at least 50 confirmed fraud cases in the SIU file; below that, the binary classifier component does not have enough signal to beat the one-class baseline.

133 tests pass. The test suite includes validity checks on the p-values themselves: given a calibration set with a known nonconformity distribution, do the p-values on a held-out genuine set have a distribution consistent with uniform? This is the fundamental correctness test, and it runs in the CI pipeline.

---

## What this does not do

It does not remove the need for claim-level investigation. The FDR guarantee is on the referral list, not on individual claims. A claim with `p = 0.003` should go to SIU; whether the investigation confirms fraud is a separate question.

It does not replace the features and model engineering required to make the anomaly detector sensitive to the fraud patterns you actually see. IsolationForest with raw claim features will give you valid p-values that are also weak. The statistical framework is only as powerful as the nonconformity scores going in.

It does not handle the case where fraud prevalence is very high: if 30% of your book is fraudulent, the null hypothesis underlying the p-value calculation (that calibration and test genuine claims are exchangeable) needs careful thought about what "genuine" means for calibration purposes.

---

## See also

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing.html) — the companion library for coverage-guaranteed prediction intervals on pricing models
- [Distribution-Free Solvency Capital from Conformal Prediction](/2026/03/11/conformal-scr-solvency.html) — applying conformal methods to SCR estimation
- [Measuring Fairness in Insurance Pricing](/2026/03/10/insurance-fairness-diag.html) — statistical tests for protected characteristic effects in rating models

GitHub: [github.com/burning-cost/insurance-conformal-fraud](https://github.com/burning-cost/insurance-conformal-fraud)
