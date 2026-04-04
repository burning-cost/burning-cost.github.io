---
layout: post
title: "Conformal Fairness When the Protected Attribute Is Missing"
date: 2026-04-02
categories: [machine-learning, regulation, fairness]
tags: [conformal-prediction, fairness, missing-data, ecj-test-achats, gdpr, consumer-duty, insurance-conformal, insurance-fairness, mask-conditional, kernel-smoothing, arXiv-2504.12582]
description: "Kong, Liu & Yang prove that standard conformal coverage guarantees degrade unevenly when protected attributes are absent at test time. With post-ECJ gender prohibition and GDPR data minimisation, UK insurers face exactly this problem. The paper closes the theoretical gap; we explain what it means in practice."
author: burning-cost
---

UK insurers face a regulatory paradox that sits at the intersection of two legal obligations pulling in opposite directions.

Consumer Duty (PS22/9) requires demonstrating fair outcomes across all groups of customers, including protected characteristics under the Equality Act 2010. The FCA's supervisory expectations are clear: you must be able to show that your models do not produce systematically worse outcomes for identifiable groups.

The ECJ Test-Achats ruling (effective December 2012) prohibited using gender as a pricing factor. GDPR data minimisation means you should not collect what you do not need. The result: the most obviously sensitive protected attribute in insurance pricing is not merely legally sensitive. It is genuinely absent from most live pricing databases.

You cannot audit fairness across a group whose membership you do not hold.

Kong, Liu & Yang (arXiv:2504.12582) address exactly this problem from the conformal prediction side: what happens to coverage guarantees when protected attributes — or any covariates — are missing at test time? The answer is more troubling than the standard treatment suggests.

---

## The coverage gap nobody talks about

Standard conformal prediction gives marginal coverage guarantees: P(y ∈ C(X)) ≥ 1−α. Under exchangeability, this holds regardless of the non-conformity score you choose. What the standard treatment does not address is what happens when some covariates are missing for a subset of the population.

Consider a fraud flagging model where 40% of records have missing occupation data — not random missingness, but MNAR: customers who are more likely to be committing fraud disproportionately decline to provide it. Standard conformal prediction calibrated on the full dataset might achieve 90% coverage overall. For complete-data records, coverage is 94%. For missing-occupation records, it is 83%.

That 7-11 point gap is not noise. It is systematic under-coverage of the group whose missingness pattern is informative. The model's uncertainty estimates are least reliable precisely for the claimants you should be most uncertain about.

The same logic applies anywhere protected attributes are absent. If younger customers are less likely to disclose disability status, and disability correlates with claims patterns, your conformal intervals are calibrated on a population whose covariate structure differs from the deployment population in a structured way.

---

## What the paper proves

Kong, Liu & Yang work through three missing data mechanisms: MCAR (missingness independent of everything — the easy case), MAR (missingness depends on observed covariates — manageable), and MNAR (missingness depends on the missing value itself — the hard case that matters most in insurance).

The marginal coverage result is reassuring: if you impute missing values and apply standard split conformal, marginal coverage is preserved under MCAR and MAR. But marginal coverage is the wrong target if your problem is fairness. You do not want coverage to hold on average. You want it to hold separately for each missingness pattern — for complete-data customers and incomplete-data customers alike.

The paper delivers two methods for this stronger guarantee:

**Nonexchangeable conformal prediction.** This extends the standard split conformal construction to achieve *mask-conditional* validity — coverage holds separately for each distinct pattern of observed versus missing covariates. The "mask" is a binary vector indicating which features are present. The method satisfies both marginal validity and mask-conditional validity simultaneously.

**Localised conformal with kernel smoothing.** A novel non-conformity score based on kernel weighting over the observed covariate space. The kernel accounts for the local density of similar observations, producing tighter intervals in data-rich regions and appropriately wider intervals where data are sparse. This achieves the strongest guarantee: marginal validity, mask-conditional validity, and asymptotic conditional validity under regularity conditions.

For the MNAR case, additional structural assumptions are required (essentially, some form of selection model or instrumental variable). Those assumptions are typically unverifiable from data alone, which is an honest limitation the paper does not hide.

---

## Where this sits relative to our existing fairness work

Our [insurance-fairness library](/insurance-fairness/) implements `PrivatizedFairnessAudit` (v0.3.8), which handles the *noise* regime: the insurer collects a privatised, locally differentially private version of the protected attribute via randomised response. You get a noisy signal; the audit is still valid.

That is a different problem from the one this paper addresses.

`PrivatizedFairnessAudit` assumes cooperative data collection — the customer responds to a randomised question, you receive an LDP-perturbed answer. The Kong/Liu/Yang paper handles the *absence* regime: no attribute data collected at all. Not privatised. Not approximate. Simply absent.

These two approaches are complementary. If you can collect a privatised signal — motor policyholders answering an anonymised questionnaire at renewal, for instance — the LDP approach applies and the noise-corrupted audit is valid. If collection is not possible (gender post-ECJ, disability in most motor contexts), the absence regime is the relevant one and this paper's methods are needed.

The practical consequence for fairness auditors: before you reach for any method, you need to characterise your missing data mechanism. MCAR is rare in insurance; if customers who decline to answer a demographic question are systematically different from those who comply, you are in MAR or MNAR territory. The mask-conditional conformal methods are designed for exactly that situation.

---

## Three concrete scenarios where mask-conditional validity matters

**Claims triage.** A model assigns a fraud risk score and conformal interval to each claim. If the interval is systematically narrower for long-standing customers with complete data than for recently acquired customers with sparse records, the threshold for referring a claim for investigation creates differential treatment. You are more confident about established customers (wider net in both directions) and less confident about newer ones (tighter intervals, less likely to flag). The demographics of each group are not random.

**Underwriting decisions at the margin.** A conformal interval around predicted loss cost guides risk acceptance decisions for borderline cases. If that interval is wider for customers with incomplete records — and those customers are disproportionately from demographic groups less likely to engage with data collection — marginal risks from those groups are more frequently declined. No discriminatory intent. Clear discriminatory effect.

**Post-hoc regulatory audit.** An FCA supervisory review or internal audit under Consumer Duty requests coverage statistics disaggregated by protected group. If group membership is missing for 30% of the portfolio — as gender effectively is for any insurer compliant with post-ECJ obligations — the audit results are unreliable without methods that account for the missingness structure. A naive analysis conflates the complete-data subpopulation with the full book.

---

## What this means for insurance-conformal

The mask-conditional validity approach extends split conformal, which is already in [`insurance-conformal`](/insurance-conformal/). The nonexchangeable method is closest to our `locally_weighted.py` and `covariate_shift.py` modules, which already handle distributional shift between calibration and deployment.

A future `MissingnessAwareConformalPredictor` class would need to: accept a missingness mask at test time, route to the appropriate kernel-weighted calibration set, and apply the nonexchangeable score. The theoretical machinery from Kong/Liu/Yang is the foundation. There is no released code from the authors as of April 2026, so implementation would require working directly from the paper.

One honest limitation: kernel smoothing degrades rapidly in high-dimensional feature spaces. A standard motor pricing model with 40–60 rating factors will have severe effective sample size problems unless the mask patterns cluster into a small number of distinct types. In practice, most insurance missingness is structural — specific fields absent for specific customer segments — so the effective number of distinct masks may be manageable. But this needs testing before any production deployment.

---

## The regulatory backdrop

The Equality and Human Rights Commission's actuarial guidance and the FCA's Consumer Duty outcome-based approach do not require you to hold protected attribute data. They require you to demonstrate that your models do not produce differential outcomes. The standard response — "we cannot test fairness because we do not hold gender data" — is no longer sufficient. Regulators are increasingly asking whether you have tested for proxy discrimination through correlated variables, and whether your model's uncertainty quantification is consistent across demographic groups you cannot directly observe.

This paper provides the theoretical foundation for why standard conformal methods fail in that setting and how to close the gap. The methods are not yet in production-ready form. The theory, however, is sound, and the problem it addresses is precisely the one UK insurers face.

**Paper:** [arXiv:2504.12582](https://arxiv.org/abs/2504.12582)
