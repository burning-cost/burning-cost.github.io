---
layout: post
title: "CAT Bond Coupons and Climate Indices: A Systematic Benchmark and What It Means for UK Reinsurance Costs"
date: 2026-04-01
categories: [reinsurance, climate]
tags: [cat-bonds, ILS, reinsurance, climate-indices, ENSO, NAO, PDO, OLR, SOI, PNA, extra-trees, gradient-boosting, catastrophe, flood, wind, retrocession, Konczal-Balcerek-Burnecki-2025, arXiv-2512-22660, UK-personal-lines, capital-cost]
description: "Kończal, Balcerek and Burnecki (arXiv:2512.22660, December 2025) systematically benchmark eight climate oscillation indices as ML features for predicting CAT bond coupon spreads across 734 bond tranches from 1997–2020. OLR, SOI and PNA matter most. Extremely randomized trees with climate variables achieves RMSE 0.012294 against a benchmark of 0.014161 without them — a 13% improvement. We explain what that means for a UK pricing actuary who has never touched an ILS product."
author: burning-cost
---

Catastrophe bond pricing is not something most UK personal lines actuaries follow closely, and that is understandable. The instruments live in capital markets, the buyers are hedge funds and pension allocators, and the perils are almost entirely US wind and US earthquake. But the coupon on a CAT bond is a price signal about catastrophe risk, and that signal travels upstream into the retrocession market, into your reinsurers' cost of capital, and eventually into the terms they offer at your annual renewal.

Kończal, Balcerek and Burnecki (arXiv:2512.22660, submitted December 2025, Wrocław University of Science and Technology) have produced what appears to be the first systematic benchmark of climate oscillation indices as features in ML models for CAT bond coupon prediction. The question they are asking is direct: among the eight major climate variability indices — ENSO, NAO, PDO, OLR, SOI, PNA, AO, and AMO — which ones actually predict the spread a catastrophe bond pays above the risk-free rate, and by how much?

It is a narrow, tractable research question. The answer turns out to matter.

---

## The dataset

734 CAT bond tranches issued in the primary market between June 1997 and December 2020. Geographic split: 57.2% United States, 7.4% Europe, 6.1% Japan, with the remainder spread across multi-regional and other single-country exposures. Peril exposure is heavily US-weighted — US wind coverage on 62.5% of tranches, US earthquake on 55.5%, multiperil on 55.6%. European wind and European earthquake appear at much lower frequencies.

This is not a balanced dataset, and the authors do not pretend it is. The US domination reflects the actual structure of the CAT bond market since its inception in the late 1990s. European perils are systematically underrepresented, which has direct implications for how usable these results are in a UK flood or European windstorm context. More on that below.

The dependent variable is the coupon spread — the yield above LIBOR (or equivalent risk-free rate) that the bond pays to investors in exchange for bearing catastrophe risk. This is the market's pricing of that risk. When a major hurricane season raises projected losses, spreads widen. When capital floods into ILS markets hunting yield, spreads compress. The paper treats this spread as the signal to predict.

---

## The models

Five ML approaches against OLS as baseline:

- Ordinary least squares (baseline)
- Random forest
- Gradient boosting
- Extremely randomized trees (ExtraTrees)
- Extreme gradient boosting (XGBoost)
- LightGBM
- Bayesian Ridge
- Automatic Relevance Determination (ARD)

Each model was run in two configurations: a benchmark set of standard bond characteristics (expected loss, rating, term, maturity, peril type) and an extended set that added the eight climate indices as additional features, with lags explored up to 12 months.

---

## What the climate indices add

The headline result is in this table from the paper. RMSE on out-of-sample predictions:

| Model | Without climate | With climate |
|---|---|---|
| OLS | 0.01844 | 0.01682 |
| Random Forest | 0.01884 | 0.01756 |
| Gradient Boosting | 0.01916 | 0.01776 |
| **Extremely Randomized Trees** | **0.01416** | **0.01229** |
| XGBoost | 0.01849 | 0.01816 |
| LightGBM | 0.01909 | 0.01886 |
| Bayesian Ridge | 0.01845 | 0.01681 |
| ARD | 0.01771 | 0.01709 |

Climate variables improve every single model. The gains are not uniform. OLS and Bayesian Ridge see roughly 9% RMSE improvement. Gradient boosting and Random Forest see around 6–7%. The two gradient boosting implementations from the major ML libraries — XGBoost and LightGBM — see only 2–3% improvement, which is surprisingly modest given their reputation for handling tabular data with mixed feature types.

The standout result is Extremely Randomized Trees. The baseline RMSE of 0.01416 is already better than every other model's *extended* configuration except its own. Add climate variables and you reach 0.01229 — a 13% improvement over its own baseline, and 27% better than the next best model's extended configuration (Bayesian Ridge at 0.01681). This is not a close race.

Why ExtraTrees rather than gradient boosting? The paper does not fully explain this, but the mechanism is plausible. ExtraTrees introduces randomness at the feature split threshold level — rather than searching for the optimal split threshold, it draws random thresholds and picks the best among them. This additional stochasticity reduces overfitting on the noisy correlations between lagged climate indices and spreads, whereas the sequential boosting approach in XGBoost can overfit to spurious lags in training data. The underlying signal from climate indices is real but weak, and high-variance models are sensitive to which lags are noise versus signal.

---

## Which climate indices matter

The correlation analysis shows a clear hierarchy. OLR (Outgoing Longwave Radiation), SOI (Southern Oscillation Index), and PNA (Pacific–North American pattern) show the strongest and most consistent positive correlations with CAT bond coupons, particularly at lags of 10 months or more. PDO (Pacific Decadal Oscillation) and ONI (Oceanic Niño Index) show negative correlations.

The ENSO story is internally consistent. SOI and ONI are both ENSO-related — SOI measures the pressure gradient between Darwin and Tahiti, ONI measures sea surface temperature anomalies in the Niño 3.4 region. Both behave similarly in the correlation analysis: negative correlation with coupons. This means La Niña conditions (negative SOI, negative ONI) are associated with higher spreads, and El Niño with lower spreads. The physical mechanism is coherent: La Niña years see enhanced Atlantic hurricane activity and reduced Pacific hurricane activity. More Atlantic hurricane risk pushes CAT bond spreads wider.

OLR's strong positive correlation is the more interesting finding. Outgoing Longwave Radiation is a measure of tropical convective activity — higher OLR indicates suppressed convection, which correlates with drier, less stormy conditions. Higher OLR predicts higher spreads — counterintuitive at first glance, but the lagged structure matters. The paper uses lags of up to 12 months, so OLR in period T is being correlated with spreads in period T+10 or T+12. The exact physical pathway is not fully elaborated.

NAO's role is more modest. The North Atlantic Oscillation showed positive correlations at shorter lags (3–5 months) but did not emerge as a dominant predictor at longer lags. Given that NAO is the primary driver of European wind variability — positive NAO phases bring stronger westerlies and more intense storms across the British Isles and northern Europe — its weaker signal is partly explained by the dataset's US dominance. The signal that NAO carries about European wind risk is swamped by the noise from 57% US-weight tranches that NAO is largely irrelevant to.

The paper does not report feature importance values from the ExtraTrees model directly. We would want to see Shapley values or permutation importance broken out by lag to understand which specific indices are driving the 13% improvement, rather than treating the climate block as a collective addition. That analysis is absent.

---

## The methodological weaknesses

The paper is competent and the question is well-framed, but there are three significant gaps.

**No peril stratification.** The most useful version of this analysis would separate US wind tranches from US earthquake from European wind and run the feature importance analysis within each peril group. OLR and SOI correlations with US hurricane activity are well-established in the meteorological literature. Whether those same indices carry signal for earthquake CAT bonds is much less clear — seismic activity is not meaningfully correlated with ocean temperature anomalies at 12-month lags. Pooling across perils muddies the interpretation. The result that climate indices improve all models could partly reflect within-sample heterogeneity: the model learns that certain market conditions (measured by climate indices) correlate with which peril mix was active in each period, rather than learning that climate indices genuinely affect individual peril pricing.

**The coverage cutoff at 2020.** The dataset ends in December 2020. The ILS market has changed substantially since then. Inflation's impact on loss adjustment costs, post-Ian repricing in 2022–23, the growing secondary market, and the inflow of new capital after the 2022-23 hard market have all altered the spread dynamics. Whether the climate-spread relationships identified in pre-2020 data hold in the current market is an empirical question the paper cannot answer.

**No out-of-time validation.** The train/test split is not described in detail in the abstract. For time-series financial data, random splits produce spuriously good results because adjacent observations are correlated. A proper evaluation would hold out the final 24 or 36 months of data as a test set and train only on prior data. Without knowing the exact split methodology, the RMSE values cannot be fully trusted.

These are not fatal flaws. The finding that climate indices carry signal for CAT bond spreads is robust to all three issues — even a diluted signal across mixed perils with a potentially optimistic split is a non-trivial finding. But the precise magnitude of the 13% improvement should be treated as an upper bound rather than a point estimate.

---

## Why this matters for UK personal lines pricing

The connection between a CAT bond coupon on a Florida hurricane deal and what Aviva charges for flood cover on a Victorian terrace in Sheffield is not obvious, but it is real. Here is the chain.

**CAT bonds set the floor on retrocession.** Catastrophe reinsurance capacity has three layers. Primary carriers buy proportional and non-proportional reinsurance from the major reinsurance groups — Munich Re, Swiss Re, Hannover Re, Scor. Those reinsurers then lay off their peak exposures into the retrocession market. The ILS market — CAT bonds, industry loss warranties, collateralised reinsurance — is the primary source of retrocession capacity for US wind and earthquake peak zones. When CAT bond spreads widen (investors demand more yield to bear catastrophe risk), retrocession becomes more expensive. When retrocession becomes more expensive, large reinsurers' cost of providing capacity for catastrophe-exposed portfolios rises. Some of that cost flows through to primary cedants at the next renewal.

**The link to UK flood and storm.** UK personal lines flood exposure is modest relative to US hurricane, but it is growing. Post-2007 summer flooding, post-2013 winter storms, and the increasing frequency of named storms hitting Ireland and western Britain have made UK flood and storm an increasingly meaningful catastrophe exposure for UK motor and household portfolios. Reinsurers who price UK flood and storm are the same counterparties who bear US hurricane risk. Their cost of capital for catastrophe exposure globally affects what they charge for UK-specific perils.

**The timing issue.** The climate index correlations in this paper operate at 10–12 month lags. ENSO phase transitions — from El Niño to La Niña and back — have cycle periods of 3–7 years. In a La Niña year, this paper suggests CAT bond spreads will be higher. Higher spreads mean ILS investors are pricing in more risk. If that feeds through into reinsurance costs at renewal, a UK insurer in a La Niña year facing its annual retrocession or XL renewal is doing so in a more expensive market. This is an empirical claim that can be tested; we are not aware of work that has specifically tracked the ENSO-reinsurance renewal lag for European perils.

**The practical implication.** A UK pricing actuary is not going to adjust their flood loading based on the current ONI reading. The transmission mechanism from climate index to reinsurance cost to primary pricing is too slow, too diffuse, and too confounded by other market factors (capacity flows, major loss events, regulatory changes) to support a direct feed. But monitoring where we are in the ENSO cycle is legitimate input to the assumption around reinsurance cost trends in your business plan — particularly for any book with meaningful flood or storm PML exposure.

If your reinsurance buyer is telling you that the market is hardening, and ENSO is currently in a La Niña phase, you have more reason to take that message seriously than if it is an El Niño year. The climate state is not the only driver of CAT bond spreads — and therefore not the only driver of reinsurance cost pressure — but it is one of them, and this paper is the first to quantify the relationship systematically.

---

## Is this directly usable by a UK actuary?

Honestly: not directly, not yet.

The model needs a time series of climate indices and a training dataset of CAT bond coupons to be useful. Even if you built the ExtraTrees model from this paper's methodology, the output is a predicted spread on a US-heavy CAT bond portfolio, not a predicted change in your XL treaty cost. The mapping from one to the other requires additional modelling that does not exist in the paper.

What the paper does give you is a principled answer to the question of which climate variables to watch if you want to understand the direction of travel in the catastrophe reinsurance market. OLR, SOI, and PNA are the primary signals. ENSO phase (via SOI or ONI) is the single most tractable indicator for a non-specialist to monitor — NOAA publishes the ONI monthly, the data is public, and the historical relationship to Atlantic hurricane activity is well-documented outside this paper's framing.

The deeper value is in establishing that the relationship is quantifiable and ML-learnable. The ILS market has suspected for years that climate oscillation phases affect spread levels, but systematic evidence has been sparse. This paper provides it. The next step — which the paper does not take — is a model that is peril-stratified, extended through 2024 or 2025, and validated on held-out years. That would be a genuinely production-useful tool for a reinsurance buyer or ILS portfolio manager.

For the UK pricing community, the honest near-term takeaway is this: climate variability is now demonstrably priced into the capital markets instruments that underpin catastrophe reinsurance capacity. Understanding which indices matter, and at what lags, is increasingly part of the background knowledge needed to intelligently interpret what your reinsurance market is doing and why.

---

## Our assessment

The Kończal-Balcerek-Burnecki paper is solid empirical work on a neglected question. The dataset is the right one to use, the benchmark is comprehensive, and the finding that climate indices improve every model tested is robust. ExtraTrees is the winning architecture, which is a non-obvious result given XGBoost's general dominance on tabular data.

The principal weaknesses — no peril stratification, no out-of-time validation, coverage ending in 2020 — are real limitations on the immediate applicability. The absence of explicit feature importance decomposition from the best-performing model is a frustrating gap. We would like to know whether OLR and SOI individually carry the majority of the climate signal, or whether the improvement is distributed across the full index set.

The reinsurance link to UK primary pricing is real but indirect. If you are attending your annual XL renewal and the market is moving against you, understanding the climate backdrop is legitimate context. It will not save you from a hard market, but it will tell you whether you are facing a cyclical correction or a structural shift.

---

**Reference:** Kończal, J., Balcerek, M. and Burnecki, K. (2025) "Machine learning models for predicting catastrophe bond coupons using climate data." arXiv:2512.22660 \[q-fin.PR\], submitted December 2025.
