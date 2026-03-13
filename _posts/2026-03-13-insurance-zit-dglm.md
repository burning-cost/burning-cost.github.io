---
layout: post
title: "Your Tweedie Model Doesn't Know About Strategic Non-Claimers"
date: 2026-03-13
categories: [libraries, tweedie, zero-inflation, pricing]
tags: [tweedie, zero-inflation, catboost, gradient-boosting, ncd, dispersion, em-algorithm, python]
description: "Standard compound Poisson Tweedie treats every zero as a Poisson draw. In UK motor, some zeros are policyholders who had incidents and chose not to claim — NCD protection, excess arithmetic. Conflating them biases both mean and zero probability. ZIT-DGLM separates the two with three-headed gradient boosting via EM."
---

Two motor policyholders. Both with a clean claims year. One is a retired teacher in Harrogate doing 3,000 miles a year in a 2018 Fiesta. The other is a courier in Manchester doing 28,000 miles. Neither claimed.

Your Tweedie model explains both zeros the same way: P(Y=0) = exp(-mu^(2-p)/(phi*(2-p))). The Poisson component drew N=0. Pure luck, or low exposure. The model adjusts accordingly — low mu for the teacher, somewhat higher mu for the courier, but both zeros treated as the same kind of event.

There is a third policyholder your model does not know to ask about: the driver in Reading with an unprotected NCD who scraped another car at a supermarket car park. Repair estimate came in at £800. Their excess is £300. Net insurance benefit: £500. But claiming drops them from 60% NCD to 30% — a premium increase of £240/year for three years. Net expected cost of claiming: £720. They did not claim. They were never going to.

That third zero is not a Poisson draw. It is a decision. And your Tweedie model cannot tell the difference.

---

## What standard Tweedie gets wrong

The compound Poisson Tweedie has P(Y=0) = exp(-mu^(2-p)/(phi*(2-p))). This comes directly from the Poisson component: N ~ Poisson(lambda), where lambda = mu^(2-p)/(phi*(2-p)). P(Y=0) = P(N=0) = exp(-lambda). Every zero in your training data is interpreted as a Poisson draw of N=0.

That interpretation is correct when every policyholder genuinely faces an insurance event with some probability, and the only reason they recorded zero is that the Poisson process happened not to fire. It is wrong when some policyholders had events and made a rational decision not to claim.

Call these structural zeros — policyholders who, given their NCD position, excess structure, and claim size expectations, will not claim regardless of whether they have an incident. Standard Tweedie conflates structural zeros with stochastic zeros (genuine N=0 draws). The consequences compound:

**The mean is biased.** When the model forces a structural zero to be explained as a Poisson draw, it reduces mu for that risk profile. But the policyholder has a real underlying claim propensity — they just exercised it strategically. The model underestimates their true risk and overestimates their zero-claim probability going forward.

**Segment-level pricing is wrong.** Non-protected NCD holders are disproportionately structural non-claimers. A model that cannot identify them will underprice adjacent risks and misprice renewal segments — precisely the segments where the excess arithmetic is most active.

---

## Where this actually happens in UK personal lines

The critical distinction is between perils where excess zeros are genuinely structural and perils where standard compound Poisson is correct.

**Motor accidental damage (own damage):** The strongest case. NCD protection is expensive — typically £30–60/year as an add-on. Without it, a first claim costs up to 65 percentage points of NCD discount. At 2024 UK premium levels (ABI: average comprehensive premium £561/year), a claim in the £400–900 range will often cost more in future premiums than the net insurance benefit. The strategic non-claiming calculation is not hypothetical. It is being run — implicitly — by millions of UK drivers on every borderline damage event.

**Home accidental damage:** Minor AD events — paint spillage, cracked tiles, broken window — are frequently absorbed without a claim because the hassle and premium impact exceed the payout. The two-stage claiming decision (has an event occurred? does the policyholder choose to claim?) is real and not modelled by standard Tweedie. Excess levels of £100–250 are low relative to many genuine AD events, but the claiming calculus still produces structural zeros for a subset of the portfolio.

**Subsidence:** Different mechanism, same model structure. Properties on clay soil, post-oak-removal, in drought-prone areas draw from a genuine Poisson process. Properties on chalk or sandstone with no tree exposure face a risk so close to zero it is effectively structural. The UK subsidence map has sharp geographic transitions — the London clay belt boundary, Jurassic clay outcrops in the Midlands — meaning the two regimes genuinely coexist within postcodes.

**Fleet motor — seasonal vehicles.** Agricultural vehicles, plant machinery, and vehicles laid up over winter are in a structural zero regime during their off-road periods. A combine harvester in a barn from November to March is not drawing from the Poisson process. Standard Tweedie assigns it positive lambda regardless.

**Where ZIT does not belong:** Windscreen. Every windscreen is exposed to stone chips; the frequency varies with mileage and road type, not a claiming decision. The windscreen excess is typically £75–100 for repair — low enough that most genuine events are claimed. Escape of water similarly: the average EoW claim is £5,000–10,000, well above any rational non-claiming threshold. Standard compound Poisson Tweedie is correct for both.

---

## The ZIT-DGLM solution

The zero-inflated Tweedie density adds a structural zero component to the standard compound Poisson distribution. For observation i:

```
f(0 | mu, phi, p, q)  = q + (1-q) * exp(-mu^(2-p) / (phi*(2-p)))
f(y | mu, phi, p, q)  = (1-q) * Tweedie_density(y; mu, phi, p)   for y > 0
```

The parameter q in [0,1] is the structural zero probability. Expected value: E[Y] = (1-q)*mu. Standard Tweedie is the special case q=0 everywhere.

The innovation in Gu (arXiv:2405.14990, May 2024) is to model all three distributional parameters — mu, phi, and q — as separate gradient boosted trees, with a generalised EM algorithm coordinating the three heads. So & Valdez (arXiv:2406.16206, NAAJ Vol 29(4):887-904, 2025) implement this with CatBoost and analytically derived custom gradients, winning the ASTIN Best Paper 2024. This is the basis of [`insurance-zit-dglm`](https://github.com/burning-cost/insurance-zit-dglm).

### The EM algorithm

The EM approach treats the structural zero indicator z_i as unobserved. For any observation, we do not know whether a zero is structural or stochastic — we compute the posterior probability of each.

**E-step:** For each zero observation, compute the posterior probability it is structural:

```
Pi_i = q(x_i) / [q(x_i) + (1-q(x_i)) * exp(-mu(x_i)^(2-p) / (phi(x_i)*(2-p)))]
```

For positive observations, Pi_i = 0 by construction — they cannot be structural zeros.

**M-step:** Update all three models using Pi_i as soft weights:
- q-model: binary cross-entropy with Pi_i as soft labels
- mu-model: weighted Tweedie deviance, downweighting likely structural zeros by (1-Pi_i)
- phi-model: weighted gamma pseudo-likelihood on unit deviances, same downweighting

Iterate until log-likelihood converges — typically 20–50 outer iterations, each running a full boosting round per head.

Each M-step reduces to a standard gradient boosting problem with modified sample weights. CatBoost handles per-head optimisation via custom `calc_ders_range` objectives. The gradients and Hessians for all three heads are derived analytically in So & Valdez (2406.16206).

---

## Why phi matters more in ZIT than in standard Tweedie

In a standard Tweedie GBM, phi is either fixed or modelled separately, but its primary function is to calibrate prediction intervals. Misspecify phi and you get wrong confidence intervals. Your point predictions — the pure premiums — are largely unaffected.

In ZIT this changes. Look at the E-step:

```
Pi_i = q / [q + (1-q) * exp(-mu^(2-p) / (phi*(2-p)))]
```

Phi appears inside the exponential that determines how much weight each zero observation receives as a structural zero. If phi is wrong, Pi_i is wrong. Wrong Pi_i means the mu-model and q-model are trained on incorrectly weighted observations. Every downstream prediction is contaminated.

To make it concrete: at mu=150, p=1.5, the Tweedie zero probability is exp(-mu^0.5/(phi*0.5)). At phi=5: exp(-12.25/2.5) = 0.0074. At phi=15: exp(-12.25/7.5) = 0.196. A factor-of-three phi misspecification changes the Tweedie zero probability by 26x. That changes Pi_i dramatically for borderline zero observations, which changes the training weights for both the mu-model and the q-model.

Fixed-phi ZIT is not a simplification. It is a cascading source of bias. The DGLM extension — modelling phi as a covariate-driven function alongside mu and q — is not optional for accurate ZIT estimation.

So & Valdez (NAAJ 2025) demonstrate this on 678,000 French MTPL policies: the ZIPB2 specification (independent q and mu with dispersion modelling) achieves Pseudo R² of 0.520 against standard Tweedie. The improvement is not from adding q alone — it is from the full three-head system.

---

## The balance problem

There is a property that standard Tweedie GLMs satisfy automatically and that ZIT gradient boosting does not. Delong and Wuthrich (arXiv:2103.03635, IME 2021) call it the balance property:

```
sum_i E[Y_i | x_i] = sum_i y_i
```

Predicted aggregate claims should equal observed aggregate claims. For a GLM with canonical link and Tweedie deviance, the zero-score equation guarantees this. For gradient boosting, there is no such guarantee.

ZIT makes the balance problem harder. The effective mean is E[Y] = (1-q)*mu — the product of two separately predicted quantities. Even if the mu-model is individually well-calibrated, systematic overestimation of q in certain segments will cause the aggregate pure premium to be understated there. The EM algorithm can converge to locally optimal (q_high, mu_high) pairs that satisfy the joint ZIT deviance but violate actuarial balance within pricing segments.

Always run a balance check before deploying.

```python
from insurance_zit_dglm import ZITModel, check_balance

model = ZITModel(tweedie_power=1.5, em_iterations=20, n_estimators=100)
model.fit(X_train, y_train)

balance = check_balance(model, X_test, y_test, groups=["region", "vehicle_age"])
# Returns overall ratio sum(E[Y_hat]) / sum(y), plus breakdown by group
```

If the ratio is off 1.0 by more than 2% overall or 5% in any material segment, apply the Delong-Wuthrich recalibration step: fit an isotonic regression layer on `log((1-q)*mu)` against actuals, then post-process predictions through it. This restores balance without re-fitting the full ZIT-DGLM.

One initialisation detail: start the EM with q_0 = fraction of zeros above the compound Poisson prediction, not the raw zero fraction. Using the raw zero fraction conflates stochastic and structural zeros from iteration zero, and the EM will converge to a locally suboptimal decomposition.

---

## Using the library

```python
from insurance_zit_dglm import ZITModel, ZITReport, check_balance

model = ZITModel(
    tweedie_power=1.5,
    em_iterations=20,
    n_estimators=100,
    link_scenario='independent',  # separate mu and q trees
)
model.fit(X_train, y_train)

# Three-headed prediction
components = model.predict_components(X_test)
# columns: mu, phi, q, E_Y

# Pure premium directly
pure_premium = model.predict(X_test)

# Full zero probability: structural + stochastic Poisson component
pr_zero = model.predict_proba_zero(X_test)  # q + (1-q)*CP_zero

# Balance check
balance = check_balance(model, X_test, y_test, groups=["region", "vehicle_age"])
```

The three-headed output is where the library earns its place. For a non-protected NCD policyholder in a segment with high strategic non-claiming, q will be elevated, mu will reflect true underlying risk, and E_Y = (1-q)*mu will be the correct pure premium. For a windscreen or EoW model — perils where the structural zero mechanism is absent — q will converge toward zero and the model reduces to standard Tweedie.

### Comparing against standard Tweedie: the Vuong test

The standard statistical test for non-nested model comparison is the Vuong test. So & Valdez use it throughout NAAJ 2025 to validate ZIT over standard Tweedie on each dataset. Run it before adopting ZIT for any peril.

```python
from insurance_zit_dglm import ZITReport

report = ZITReport()
result = report.vuong_test(model_zit, model_tweedie, X_test, y_test)
# VuongResult: test_statistic, p_value, preferred_model
```

If the Vuong test does not reject standard Tweedie for your data — particularly in low-zero-inflation perils — do not force ZIT. The compound Poisson model is right for those lines.

---

## Where this sits in the Burning Cost stack

[`insurance-poisson-mixture-nn`](https://github.com/burning-cost/insurance-poisson-mixture-nn) handles structural vs stochastic zero decomposition in claim frequency — a PyTorch neural network for count data. The relationship with ZIT-DGLM is complementary, not competitive.

**PM-DNN** models claim counts — P(N=0), P(N=k). It operates on the frequency dimension and works best with telematics data, where mileage and driving behaviour separate structural from stochastic zeros at the individual level. PyTorch, count data, frequency only.

**ZIT-DGLM** models aggregate claims — P(Y=0), E[Y], the full semi-continuous loss distribution. It operates on aggregate losses directly, like standard Tweedie, and adds covariate-driven dispersion. CatBoost, continuous semi-continuous data, pure premium.

For a telematics motor pricing stack: PM-DNN handles frequency components and ZIT-DGLM handles the aggregate loss model. Run both. If PM-DNN assigns high structural zero probability to a segment where ZIT-DGLM's q is near zero, you have a consistency failure worth investigating before either model goes near a rating engine.

---

## The regulatory angle

FCA Consumer Duty (PRIN 2A, July 2023) requires pricing to produce fair value and avoid systematically overcharging lower-risk customers. A standard Tweedie model that absorbs strategic non-claimers into a low-mu bucket — and then rewards them with renewal discounts for their clean history — creates a systematic cross-subsidy. The policyholder who rationally chose not to claim a £600 accidental damage incident to protect their NCD is not the same risk as the policyholder who had no incidents. Pricing them identically is a fair value failure with a specific and traceable mechanism.

ZIT-DGLM does not fix this automatically. But q(x) — the structural zero probability as a function of covariates — is auditable, explainable, and documentable in a model card in a way that "we applied Tweedie and the zeros sorted themselves out" is not. The segment-level balance check gives you the actuarial evidence that the model is not systematically cross-subsidising between NCD segments.

---

## What we think

Standard Tweedie GBMs are right for most UK personal lines perils, most of the time. For windscreen, EoW, liability — the compound Poisson mechanism is real and the zero proportion is explained by the Poisson rate. Do not use ZIT where the data does not require it.

For motor accidental damage with unprotected NCD, home accidental damage, subsidence, and commercial fleet with seasonal vehicles — the structural zero argument is not theoretical. At current UK premium levels, the rational non-claiming threshold for unprotected-NCD holders sits between £350 and £850 for a typical motor policy. Incidents in that range are occurring and not being claimed. Your current Tweedie model treats all of them as Poisson draws of N=0. It is wrong in a way that matters for pricing.

ZIT-DGLM gives you the language to model this properly. The DGLM extension — separately modelling phi as a covariate-driven quantity — is what makes it a complete solution rather than a patched one. And the balance check is what makes it actuarially deployable rather than academically interesting.

```bash
uv add insurance-zit-dglm
```

---

*Papers: Gu (2024) arXiv:2405.14990 — So & Valdez (2024) arXiv:2406.16206 — So & Valdez (2025) NAAJ Vol 29(4):887-904 — Delong & Wuthrich (2021) arXiv:2103.03635, IME 2021*

---

**Related reading:**
- [Separating Structural Non-Claimers from Risk: Mixture Cure Models for Insurance Pricing](/2026/03/11/insurance-cure/) — mixture cure models as an alternative: parametric separation of the zero-inflation probability from the severity model, with the cure fraction having a structural interpretation
- [Double GLM for Insurance: Every Risk Gets Its Own Dispersion](/2026/03/11/insurance-dispersion/) — joint modelling of mean and dispersion; the DGLM precursor to the full zero-inflated DGLM, for data without structural zeros but with heterogeneous variance
- [GAMLSS in Python, Finally](/2026/03/10/insurance-distributional-glm/) — the broader framework: every distributional parameter, including the zero-inflation probability, as a function of covariates; GAMLSS generalises both DGLM and the ZI-DGLM
