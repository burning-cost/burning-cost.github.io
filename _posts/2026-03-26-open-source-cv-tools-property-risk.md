---
layout: post
title: "Open-Source Computer Vision Tools for Property Risk: What Is Actually Usable"
date: 2026-03-26
categories: [property-insurance, computer-vision]
tags: [computer-vision, satellite-imagery, segment-anything, roboflow, open-source, property-risk, underwriting, aerial-imagery, uk-insurance, pricing]
description: "SAM2, GroundingDINO, YOLOv8, Roboflow Universe — the open-source CV stack for building analysis exists. This is an honest account of what it can do, what training data you actually need, and where it falls apart for UK property risk."
---

The commercial aerial analytics vendors — Cape Analytics, Verisk, Nearmap — run proprietary computer vision pipelines trained on millions of labelled property images. If you are a UK insurer without access to those products, the obvious question is whether the open-source ecosystem gives you a credible alternative. It does, partially, and with a harder build than the vendor demos suggest.

This is the second post in our series on computer vision and LLMs for property insurance. The [first post](/2026/03/26/computer-vision-property-insurance-what-actually-works/) covered the deployed vendor landscape, UK data infrastructure, and the regulatory position. This one goes into the open-source tooling: what exists, what it actually does, and where the gaps are.

---

## The foundation model tier

The segmentation and object detection landscape changed materially between 2022 and 2025. You no longer need to train a model from scratch to get usable building segmentation.

**Segment Anything 2 (SAM2)**, released by Meta in 2024, is the current state of the art for zero-shot instance segmentation. Feed it a property image — aerial or street-level — and it will segment objects without any task-specific training. For roof boundary delineation from above, SAM2 performs credibly. The limitation is that segmentation is not classification: SAM2 will draw a polygon around the roof but will not tell you the material is concrete tile versus slate versus felt-over-fibreboard, nor whether there is moss or visible defect. Classification requires either a separate model or a multimodal LLM on top of the segmented region.

**GroundingDINO** (IDEA Research, 2023) couples an object detector to a language model, letting you specify detection targets in text: "cracked render", "flat roof extension", "solar panel array". Zero-shot detection performance on general objects is strong. On property-specific features — subtle roof condition indicators, the difference between a conservatory and a single-storey kitchen extension — it degrades. Grounding DINO was trained on broad object detection benchmarks, not on building condition data. It will find the obvious features; it will miss the diagnostic ones.

**AutoDistill** (Roboflow, 2023–ongoing) chains GroundingDINO and SAM together as a label generation pipeline. The concept: describe the classes you want, run AutoDistill over unlabelled imagery, get a weakly-labelled training set, fine-tune a small detector. In practice this reduces manual labelling effort significantly. We would estimate a 60–70% reduction in annotation time for a standard roof condition classification task — not elimination of annotation, but meaningful reduction.

---

## YOLOv8 and the practical detection pipeline

For deployment rather than experimentation, **YOLOv8** (Ultralytics, 2023) remains the default choice. Fast inference, well-documented fine-tuning API, works well on aerial imagery at sub-20cm resolution. YOLOv9 and YOLOv10 have since been released with architectural improvements, but the practical performance difference on property imagery is small enough that it rarely justifies migrating an existing pipeline.

The standard pipeline for a roof condition classifier looks like this:

1. Collect aerial imagery tiles (Vexcel commercial, or OS MasterMap Imagery, or aerial photography from a provider like Getmapping)
2. Segment individual roof polygons using SAM2 or Detectron2 (Meta's Mask R-CNN implementation)
3. Crop roof patches to standardised bounding boxes
4. Label condition categories — typically Excellent / Good / Fair / Poor / Severe — against ground truth
5. Fine-tune YOLOv8 or a lightweight ResNet on the labelled crops
6. Validate against held-out labelled data; ideally cross-validate against claims outcomes

Steps 1, 2, and 5 are well-supported by open tooling. Steps 4 and 6 are where UK efforts would stall, and we cover that below.

**SAHI** (Slicing Aided Hyper Inference) is worth knowing about if you are working with high-resolution aerial tiles. Large-format images contain small features — a cracked ridge tile is a handful of pixels in a 7cm resolution tile — and standard detection models miss them because they are trained on normalised input sizes. SAHI slices the image into overlapping sub-windows before inference and merges the results, substantially improving detection of small features. This is the right approach for property defect detection at aerial resolution.

---

## Roboflow Universe: what is actually there

**Roboflow Universe** is a collaborative dataset and model repository with over 250,000 public datasets. The property-relevant content is thinner than it looks.

The most relevant item for this use case: a roof detection dataset with approximately 280 annotated rooftop images and a pre-trained detection model. That is not a large dataset by the standards of any industrial CV application. For comparison, Cape Analytics v5 was trained on data derived from 20 million exposure records; the Roboflow roof dataset is roughly 70,000 times smaller by exposure volume.

The 280-image model will detect roof polygons in clear aerial imagery reasonably well. It will not produce calibrated condition scores. It will not generalise to UK building stock: it was assembled from US residential imagery (detached, suburban, predominantly asphalt shingle), which has limited visual similarity to UK terrace housing, Victorian brick semi-detached, or flat-roof extensions on bungalows.

There are also **xView2** and **xBD** datasets (DARPA-funded building damage assessment, post-disaster imagery) and multiple rounds of **SpaceNet** challenges on building footprint extraction. These are useful for the computer vision pipeline — segmentation, detection — but the label schema is "damaged vs undamaged post-event", not "roof condition for underwriting". The transfer task is real and requires domain adaptation.

---

## Geospatial tooling

For working with georeferenced imagery — which is what you get from Vexcel, EA LiDAR, OS MasterMap — the standard Python stack is:

- **Rasterio** for reading and writing raster data (GeoTIFF, COG)
- **GeoPandas** for vector data (building footprints, polygon overlays)
- **GDAL** underneath both; rarely interact with directly but worth having functioning
- **Shapely** for geometry operations (intersection, buffering, centroid extraction)

This stack is mature, well-documented, and straightforward to install. The gotcha is coordinate reference systems: Vexcel imagery may be delivered in British National Grid (OSGB36 / EPSG:27700) while your pricing model expects WGS84 latitude/longitude. Mixing CRS is responsible for a disproportionate share of geospatial pipeline bugs.

The **satellite-image-deep-learning** GitHub repository (maintained by the community since 2019) is the best single index of techniques, datasets, and pre-trained models for this domain. If you are starting a property imagery project, read this before building anything.

---

## The multimodal shortcut: GPT-4o on property photos

The alternative to building a classical CV pipeline is to pass property images directly to a vision-language model and extract structured features via prompt engineering.

GPT-4o (OpenAI, 2024) can, from a street-level or aerial property image, describe:
- Roof material and apparent condition
- Observable extensions, conservatories, flat roof additions
- Solar panel presence
- Vegetation proximity (wildfire/storm damage risk proxy)
- Property type (detached / semi-detached / terraced)
- Approximate age from architectural style

The output is a natural language description that can be parsed into structured fields. Cost is approximately $0.01–0.05 per image at current API rates (a typical property assessment prompt with one image sits in the $0.015 range). For a UK portfolio of 1 million properties, a one-time enrichment pass would cost roughly £12,000–£40,000 at current rates — feasible for a pilot.

Open-source alternatives include **LLaVA** (various versions; Llama 3.2 Vision is the current strong open-weight option), **InternVL2**, and **Qwen-VL**. These can run locally — relevant if imagery data governance prevents sending property images to a third-party API. Performance is meaningfully below GPT-4o on nuanced property assessment tasks, but adequate for binary classification (flat roof present / absent, solar panels present / absent, extension visible / not visible).

The honest limitation: these models are describing what they see in natural language. They are not producing calibrated probability estimates. "The roof appears to be in fair-to-poor condition with visible moss growth along the ridge" is not the same as a validated roof condition rating. Converting that description into a numeric score that can enter a GLM pricing model requires a further layer: either human review or a secondary classification model trained on the LLM's outputs against ground-truth outcomes. The pipeline is longer than it appears on initial implementation.

---

## Where the open-source stack falls short for UK property insurance

We want to be direct about the gaps, because they are not minor.

**No UK-specific training data exists in the open literature.** Every published roof detection or building condition dataset we found is US-sourced or globally aggregated (SpaceNet covers multiple cities but US-heavy). UK building stock — the Victorian terrace with bay window, the 1930s semi-detached, the flat-roof extension, the thatched cottage — is visually distinct from US suburban housing. A model trained on US data and evaluated on UK imagery will degrade; by how much is unquantified because there are no public UK benchmarks.

**No ground-truth labelling connecting imagery to UK insurance outcomes.** The Cape Analytics 700,000-claim validation is precisely the dataset that gives their roof condition scoring its commercial value. To replicate that validation for a UK model, you would need: matched aerial imagery for each policy (at the same resolution and date as underwriting), policy characteristics, and the subsequent claims record. No UK insurer has published that they hold this dataset. Building it requires imagery procurement (Vexcel or Getmapping commercial licence), address matching to UPRN, and time-alignment of image date to policy exposure — a substantial data engineering project before any model training begins.

**The annotation problem is underestimated.** A binary roof / not-roof detector requires minimal labelling. A calibrated five-class roof condition model (Excellent / Good / Fair / Poor / Severe) requires consistent annotation by people who understand what those condition categories mean in the context of property claims — not generic image labellers. Industry knowledge needs to be encoded into the label schema. That is slow and expensive.

**Cloud cover degrades UK aerial imagery reliability.** The UK maritime climate means optical imagery — the kind that works for visual condition assessment — may have only two to four cloud-free acquisitions per year for some regions. Your training dataset and your production scoring need to account for image quality variance. A model that performs well on clear-day summer imagery may fail on images taken under diffuse overcast light, which is a high proportion of available UK imagery.

**SAR does not substitute.** ICEYE and Sentinel-1 penetrate cloud cover but SAR imagery requires specialist interpretation. The visual features that indicate roof condition — colour changes, texture, moss, cracking — are not visible in SAR backscatter. SAR is the right tool for flood extent and terrain; it is not a replacement for optical imagery for building condition assessment.

---

## The realistic scope for a UK team

Given all of the above, what can a UK pricing or data science team actually build with open-source tooling in 2026?

The most defensible near-term applications, ranked by feasibility:

**1. Binary attribute extraction from Street View.** Identifying flat roof extensions, solar panels, obvious property type mismatches against declared type — these are binary classification tasks that can be tackled with LLM zero-shot or fine-tuned YOLOv8 on a few hundred labelled examples. The data quality bar is lower for binary tasks. The Johansson et al. (2026, arXiv:2601.06056) paper demonstrated exactly this kind of zero-shot LLM classification at scale (154,710 Swedish buildings from Street View for heritage identification). The same approach is directly applicable to UK non-standard construction flags.

**2. Spatial embeddings for sub-postcode territory rating.** The Holvoet et al. (2025, arXiv:2511.17954) contrastive learning approach — combining satellite imagery tiles with OpenStreetMap features to produce lat/lon embeddings — is probably the highest-leverage open-source application for a UK pricing team right now. It does not require building-level condition labels. It requires training compute and imagery access (Sentinel-2 is free at 10m resolution) but the training task is self-supervised. The embeddings slot directly into a GLM or GBM as additional covariates. Post-training, scoring a new location requires only coordinates, not fresh imagery. We would prioritise this above any property-level condition scoring project given the current data constraints.

**3. EPC enrichment via address matching.** Not CV at all, but worth stating: the 30 million EPC records available free from MHCLG contain ROOF_DESCRIPTION, WALLS_DESCRIPTION, and CONSTRUCTION_AGE_BAND fields that are almost certainly unused by most UK home insurers at scale. Matching these to a pricing portfolio by UPRN is a data engineering task, not a machine learning task. The coverage gap (65% of properties, new builds over-represented) limits this, but for the covered subset it provides structured building attributes without any imagery processing.

**4. Proof-of-concept roof condition scoring.** Feasible with Vexcel imagery (commercial licence required), SAM2 for roof segmentation, GPT-4o or fine-tuned YOLOv8 for condition classification, and human expert review to establish ground truth. This is a 3–6 month project to get to validated pilot stage, not a weekend hackathon. Do not attempt this without securing the imagery licence and having a claims validation dataset in scope from the start.

---

## What to actually build first

The mistake we see teams making is starting with the interesting problem — automated roof condition scoring — before establishing the data infrastructure. The right order is: get imagery under licence and understand its quality and refresh cycle; match to UPRN; establish a validation dataset with claims; then train a model. Skipping ahead to model training on publicly-available US data and hoping it transfers to UK conditions is how you spend six months and produce a model that cannot be deployed.

The open-source stack — SAM2, YOLOv8, Rasterio, AutoDistill — is good. It removes the computer vision barriers that would have made this project infeasible five years ago. The remaining barriers are data (UK ground truth), infrastructure (imagery refresh at useful cadence), and governance (proxy discrimination characterisation before deployment, covered in the [first post](/2026/03/26/computer-vision-property-insurance-what-actually-works/)). Those are not problems that better open-source tooling will solve. For proxy discrimination governance of any imagery-derived feature, [`insurance-fairness`](/insurance-fairness/) is the right tool before production deployment.

---

*The third post in this series covers LLMs for the operational layer of property insurance: text extraction from surveys and loss adjuster reports, structuring unstructured property data, and the hallucination problem in claims contexts. It is a different application to the imagery pipeline covered here, though the tools overlap.*
