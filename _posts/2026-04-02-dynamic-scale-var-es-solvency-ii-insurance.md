---
layout: post
title: "Dynamic Scale Models for Solvency II: Interesting for Banks, Less So for Insurers"
date: 2026-04-02
categories: [techniques, capital, solvency-ii, actuarial]
tags: [var, expected-shortfall, solvency-ii, scr, garch, quantile-regression, dynamic-scale, reserving, capital-modelling, market-risk, liu-luger, arXiv-2603-02357]
description: "Liu & Luger (arXiv:2603.02357) build a semiparametric VaR/ES forecaster that models scale dynamics through quantile differences rather than GARCH variance equations. The method works well on equity indices. Whether it transfers to insurance loss distributions is a harder question than the paper acknowledges."
math: true
author: burning-cost
---

A paper from Liu and Luger (arXiv:2603.02357, March 2026) proposes a semiparametric approach to forecasting Value-at-Risk and Expected Shortfall by modelling the conditional scale of financial returns as the difference between two specified quantiles. No parametric distributional assumptions. No GARCH variance equation. Outperforms standard GARCH and joint VaR-ES models on major international equity indices, including through the COVID volatility spike. It is methodologically clean work.

The problem is that equity indices and insurance loss distributions are different animals, and the paper does not meaningfully engage with that difference.

---

## What the method does

The central idea is to replace the usual GARCH conditional variance model with a "restricted quantile regression" that estimates the spread between two quantiles — say the 75th and 25th percentile of daily returns — and treats that spread as a proxy for the conditional scale of the distribution. VaR at a given confidence level is then estimated from the tail quantiles of the rescaled returns. Expected Shortfall is approximated by averaging quantiles below the VaR threshold.

This is more honest than GARCH-based ES in one important respect: it makes no assumption about the shape of the tails. If the return distribution is skewed or leptokurtic on some days and not others, the quantile-based scale adapts without forcing the data into a fixed parametric family. The authors validate this on daily return series for major equity indices — S&P 500, FTSE 100, DAX, Nikkei — and benchmark through the March 2020 COVID period.

The distributional-free framing is genuinely useful when the data-generating process is unstable. Anyone who has tried to fit a stable GARCH to equity returns during a volatility regime change will recognise the problem the paper is solving.

---

## Where Solvency II enters

The Solvency II Solvency Capital Requirement (SCR) is defined as the 99.5th percentile VaR over a one-year horizon. For internal model firms with market risk exposure — equity, interest rate, property — the question of how you estimate that VaR matters. A dynamic scale approach that captures regime changes without baking in parametric assumptions is conceptually attractive for the market risk module.

If you are a UK non-life insurer or Lloyd's managing agent running an internal model, and if your market risk component is material, this paper is worth reading. The dynamic scale idea is structurally analogous to what you want: a method that does not assume your investment portfolio volatility is stationary, because it obviously is not.

That is a real but narrow use case. Most UK non-life pricing actuaries will not touch Solvency II SCR estimation directly.

---

## The transfer problem

The paper benchmarks entirely on equity index daily returns. This is a specific data-generating process with well-documented properties: autocorrelated squared returns (GARCH effects), leverage effects (asymmetric response to positive and negative shocks), heavy tails. These properties have been studied since the 1980s. They are why GARCH was invented.

Insurance loss distributions do not reliably exhibit these properties.

Annual or quarterly reserve triangles have tens or hundreds of observations, not thousands. The autocorrelation structure is different — development periods are correlated in ways that reflect claim settlement patterns, not volatility clustering. The tail behaviour of a liability distribution in a run-off year is determined by large individual claims, legal developments, and inflation dynamics, not by the kind of temporal scale dynamics that drive equity market VaR.

The COVID test period is a genuine regime change for equity markets. But its status as a validation case for an insurance loss model is more ambiguous. The COVID period did produce real insurance losses — trade credit, BI, event cancellation — but those losses were driven by specific policy wordings and coverage disputes, not by a general increase in loss scale volatility that a dynamic scale model would have detected in advance.

This matters because the paper's strongest empirical claim — that dynamic scale outperforms GARCH during regime change — rests on a dataset where regime change has a specific, well-understood meaning. That meaning does not straightforwardly carry over to reserving or pricing.

---

## What would change our mind

We would want to see this method applied to reserve risk distributions: either to the empirical distribution of reserve run-off ratios, or to paid loss development triangles across accident years. Do scale dynamics in reserve triangles predict subsequent reserve deterioration? Is there empirical evidence of volatility clustering in loss ratios? If yes, the paper becomes significantly more interesting for reserving actuaries.

The ODP bootstrap, which most UK reserving teams use as their reserve risk distribution workhorse, does not model temporal scale dynamics at all. If Liu and Luger's approach outperforms ODP bootstrap out-of-sample on reserve triangles, that is a publishable result and a practical argument for adoption. But that test has not been run.

For the SCR market risk module, the case is stronger and does not need additional validation — the equity index data is the right data for market risk. The question there is whether your internal model is sophisticated enough that swapping in a dynamic scale approach for GARCH would pass the Use Test.

---

## The practical toolkit

Our [insurance-quantile v0.5.0](https://github.com/burning-cost/insurance-quantile) library, released this week, includes an ES regression module implementing the Patton/Taylor joint VaR-ES loss function approach. That is a different method from Liu and Luger's quantile-scale-difference framework — the implementations do not overlap — but if you are building out a capital modelling pipeline that includes ES estimation, it is the practical starting point. The Patton/Taylor approach is better documented in actuarial contexts and has cleaner calibration properties for the kind of annual loss distributions insurers work with.

---

## Our verdict

The methodology is sound and the distributional-free framing is a genuine contribution over GARCH-based ES. For banks and asset managers running equity VaR models under Basel IV or FRTB, this is directly applicable. For UK insurance capital modellers working on the market risk SCR module, it is worth awareness.

For pricing actuaries, the immediate relevance is minimal. The interesting question — do insurance loss distributions exhibit temporal scale dynamics worth modelling? — is not answered by this paper and probably requires substantially different data and analysis. We think the answer is "sometimes, in specific lines," but we would not adopt this without seeing the reserve triangle validation work done first.

The GARCH comparison the paper uses as its baseline is a strawman in insurance. Nobody uses GARCH for reserve risk in practice. The relevant baseline is the ODP bootstrap or a Mack model, and the paper does not test against either.
