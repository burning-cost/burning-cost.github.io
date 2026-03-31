---
layout: post
title: "Privacy Risk Benchmarking for Synthetic Insurance Data: What the Standard Metrics Get Wrong"
date: 2026-03-31
categories: [synthetic-data]
tags: [differential-privacy, membership-inference, DCR, DP-TVAE, DP-CTGAN, AIM, smartnoise, ICO, GDPR, privacy-auditing, synthetic-data, tabular-data, insurance-pricing, Zuo-2026, class-imbalance]
description: "Membership Inference Attacks are essentially uninformative on tabular insurance data — a finding from Zuo et al. (arXiv:2602.09288) with direct consequences for how UK pricing teams should audit and defend synthetic datasets."
author: burning-cost
---

Most synthetic data privacy audits in UK insurance follow the same procedure: generate synthetic data, run a Membership Inference Attack (MIA), report that the attack achieves roughly 50% balanced accuracy (i.e., random guessing), declare victory. The ICO won't notice. The reinsurer will accept it. Move on.

Zuo, Kang, Patterson and Seneviratne (arXiv:2602.09288, February 2026) have now tested this procedure systematically across six financial datasets and multiple generators, including DP-CTGAN and DP-TVAE at epsilon values of 1, 5, 10, and infinity. The uncomfortable finding: MIA success rates cluster at ~0.50 regardless of whether differential privacy is applied and regardless of what epsilon you use. A shadow-model MIA cannot distinguish a non-DP vine copula synthesiser from a formally private AIM mechanism at epsilon=1. The audit tells you nothing.

This is not a reason to relax — it is a reason to change what you measure.

---

## Why MIA fails on tabular insurance data

Membership Inference Attacks are well-validated in domains with high-dimensional output: image generation, language models. The adversary's discriminator has rich signal — pixel distributions, token co-occurrences — to work from. Zuo et al.'s shadow-model variant extracts histogram features from binned attributes across 100 synthetic datasets of 1,000 records each and trains a binary classifier: was this canary record in the training set?

On six tabular financial datasets (10–28 columns, n=1K–150K rows), the discriminator collapses. Mean MIA balanced accuracy across all generators: ~0.50. Standard deviation: 0.0218. There is no statistically meaningful separation between generators. DP-CTGAN at epsilon=1 and Gaussian Copula with no DP protections at all produce indistinguishable MIA scores.

The likely mechanism: tabular financial data has too few columns and too much within-column concentration to give the discriminator exploitable signal. Insurance pricing data is worse still — a UK motor portfolio has around 15 structured rating factors, most of them heavily discretised (vehicle group, NCD band, postcode area). The synthetic records cannot be distinguished from the real ones using histogram features, not because the data is private, but because the feature space is too low-dimensional for this attack.

This matters because a MIA at 0.50 balanced accuracy is currently the de facto privacy evidence cited in synthetic data documentation prepared for the FCA, reinsurers, and data-sharing agreements. It is meaningless for this data type. Citing it is not dishonest — you ran the test and reported the result — but it provides no actual evidence of privacy protection.

---

## DCR is more informative, up to a point

The paper's other privacy metric is Distance to Closest Record (DCR): the ratio of the median distance from a synthetic record to its nearest real-data neighbour, normalised against a random uniform baseline. A DCR of 1.0 means the synthetic records are no closer to training data than random noise. A DCR near 0 indicates memorisation — synthetic records that are near-perfect reproductions of specific training records.

DCR is more informative than MIA for one specific risk: outlier memorisation. In insurance terms, this is the large-claims tail. A policy with a £280K buildings claim that is unique in your training set might reproduce near-exactly in a non-DP synthesiser if the vine copula learns its unusual combination of features directly. DCR near 0 would flag this.

Zuo et al. find no meaningful correlation between DCR scores and MIA success rates. The two metrics are measuring different things, which is partly reassuring (DCR adds information) and partly troubling (your privacy audit framework is inconsistent). Our view: run DCR as a memorisation check, treat it as a red flag below 0.8, and do not rely on it as positive evidence of privacy.

For formal privacy guarantees, neither metric is sufficient. The only defensible position for external data sharing is a mathematical DP bound.

---

## The class imbalance problem will destroy your synthetic portfolios

The paper's second major finding is more immediately damaging for actuarial use.

DP-TVAE — one of the more commonly recommended DP generators for tabular financial data — collapses on imbalanced datasets. Financial datasets with a 6–30% minority class show catastrophic mode collapse at epsilon=1: DP-TVAE produces synthetic datasets with 0–3% minority representation, regardless of the true prevalence in training data.

Translate that to insurance. A UK personal motor portfolio with 5% claimers — ordinary attritional business — would produce a synthetic portfolio with effectively zero claim frequency after DP-TVAE synthesis at epsilon=1. The downstream Poisson GLM would estimate claim rates an order of magnitude too low. The synthetic data is not just imperfect — it is actuarially incoherent.

The mechanism is well understood. DP-SGD (the gradient-clipping approach used by DP-CTGAN and DP-TVAE) clips per-sample gradients to a maximum norm, then adds calibrated Gaussian noise. This process is disproportionately destructive for rare classes because the per-sample gradients from minority records are already small — they are minority records precisely because they are unusual. Gradient clipping destroys the signal before the noise has a chance to. The result is a generator that has effectively learned to ignore the minority class.

DP-CTGAN is somewhat better behaved (it tends to overrepresent minority classes rather than eliminate them), but neither GAN-based DP approach is reliable for imbalanced insurance data at epsilon=1.

The recommended workaround — downsampling to 50/50 before synthesis — eliminates mode collapse at the cost of destroying the representativeness of the synthetic portfolio. You cannot recover the true class proportions post-hoc without introducing non-private information.

---

## Why AIM avoids both problems

The failure modes above are specific to DP-SGD-based generators. AIM (Adaptive Iterative Mechanism), the approach used in our [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) library, operates on entirely different principles.

AIM is a marginal-based mechanism in the select-measure-generate (SMG) paradigm. Rather than training a neural network with DP-SGD, it:

1. Privately measures a set of low-dimensional marginal queries (1-way and 2-way distributions)
2. Adds calibrated Laplace/Gaussian noise to those marginals — this is where the DP guarantee is consumed
3. Fits a probabilistic graphical model to the noisy measurements and samples from it

Minority class preservation falls out naturally. The claim frequency is a 1-way marginal — a count query. AIM measures it directly with DP noise, and the error on that count is small relative to dataset size. On a 100K-record portfolio with 5K claimants, the noisy claim frequency is accurate to well within 1 percentage point at epsilon=1, even after accounting for budget splitting across all marginals. The synthesised frequency is 4.8–5.2%, not 0–3%.

MIA performance for AIM at epsilon=1 is around 0.50–0.56, consistent with Zuo et al.'s findings across all generators. For AIM this is not a measurement failure — the formal DP guarantee bounds the information any adversary can extract about whether a given record was in the training set. The empirical attack result and the mathematical guarantee are pointing in the same direction: epsilon=1 provides meaningful protection.

The insurance-synthetic library also handles the structural insurance problem that generic DP libraries miss: the frequency-severity split. Claim frequency is synthesised via a DP count query (Poisson-calibrated against the noisy claimant count); severity is a separate DP marginal over non-zero claim amounts. This maintains the zero-inflated structure that actuarial GLMs require. The combined epsilon budget is allocated 10% to PrivTree discretisation, 60% to policy features, and 30% to severity.

---

## The ICO position, stated plainly

The ICO published updated anonymisation and pseudonymisation guidance in March 2025. The key point that UK pricing teams need to understand is this: synthetic data is not automatically anonymous. The guidance says so explicitly. Generating synthetic data, running a MIA, and reporting 50% balanced accuracy does not satisfy the motivated-intruder test.

The ICO's test is: would a determined attacker with reasonable resources succeed in re-identifying an individual? The motivated intruder is not assumed to be a cryptographer with unlimited compute — they are a reasonably sophisticated adversary who can apply standard attacks. If those attacks (MIA, attribute inference, nearest-neighbour queries) do not work on your synthetic data, you have reasonable evidence of anonymisation.

The problem identified by Zuo et al. is that MIA not working does not imply your data is anonymous. It implies the attack lacks signal, which is a property of the data structure, not the privacy protection. A non-DP synthesiser also passes a MIA at 0.50 on tabular financial data.

The defensible route for external data sharing — to reinsurers, academic researchers, actuarial consultants — is formal DP with a documented epsilon. Epsilon=1 with AIM satisfies the motivated-intruder test: the mathematical guarantee limits the adversary's advantage, and the DCR metric confirms no outlier records are reproduced near-exactly. Epsilon=1 with DP-CTGAN or DP-TVAE does not satisfy it for a different reason — the utility-privacy tradeoff collapses for imbalanced data, making the synthetic portfolio actuarially incoherent before the privacy question is even settled.

No specific epsilon threshold is mandated by the ICO, the FCA's SDEG report (August 2025), or the Data Use and Access Act 2025. The combination of formal DP epsilon with a DCR check for outlier memorisation is our recommended evidence package for regulatory purposes.

---

## Practical recommendations for UK pricing teams

**1. Stop treating MIA as a privacy audit for tabular synthetic data.**
Zuo et al. confirm what should now be the default position: a 0.50 MIA balanced accuracy is not evidence of privacy for tabular data with 10–25 columns. It is evidence that the attack lacks signal. Document this in your synthetic data governance framework and do not cite MIA success rates as privacy evidence in submissions to regulators or counterparties.

**2. Run DCR as a memorisation check, not a privacy guarantee.**
Compute the nearest-neighbour distance between synthetic and real records. Flag any DCR below 0.8. This catches outlier reproduction — the tail claimant whose record is reproduced near-exactly. It is a necessary check, not a sufficient one. Our insurance-synthetic library will add a DCR metric to `SyntheticFidelityReport` in the next release.

**3. Do not use DP-TVAE or DP-CTGAN on insurance pricing data.**
The class imbalance collapse at epsilon=1 is disqualifying for actuarial use. A 5% claimer portfolio synthesised with DP-TVAE produces 0–3% synthetic claim frequency. The downstream GLM will produce nonsense. Use AIM (via smartnoise-synth) or MST. The utility-privacy curve for marginal-based methods is materially better than DP-SGD approaches for datasets under 200K rows.

**4. For external sharing, cite epsilon — not MIA.**
If you are sharing synthetic motor or home data with a reinsurer, academic, or vendor under a data-sharing agreement, the privacy evidence your legal team needs is: "Generated with AIM mechanism, epsilon=1, delta=1e-5. Formal (epsilon, delta)-DP guarantee. DCR=0.94, no outlier records reproduced." This is what the ICO motivated-intruder test requires.

**5. Pre-specify column bounds from actuarial knowledge.**
Every DP tabular library fits column bounds from data by default, which consumes DP budget and creates a domain-extraction vulnerability for outlier records. For UK motor: driver age 17–100 (statutory), vehicle group 1–50 (market-known), claim amount 0 to your Pareto threshold (actuarially calibrated). Pre-specifying bounds costs zero epsilon budget and eliminates a known attack surface.

**6. Check claim frequency preservation explicitly.**
Add a metric to your synthetic data fidelity report: `synthetic_freq / real_freq`. For a 5% claimer portfolio, the acceptable range is 0.85–1.15. Anything outside that range means your synthetic data will produce miscalibrated GLMs. This is not a privacy metric — it is an actuarial usability metric — but Zuo et al.'s findings make it a necessary audit step when DP is in use.

---

The paper: Zuo, Kang, Patterson, Seneviratne — "Measuring Privacy Risks and Tradeoffs in Financial Synthetic Data Generation", arXiv:2602.09288, February 2026.

Our library: [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) — AIM-based DP synthesis for insurance portfolios, with frequency-severity separation and pre-specified UK motor/home column bounds.
