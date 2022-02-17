# tf-md
### The Diagnostic Transformer
![tests](https://github.com/prob-ml/tfmdoc/workflows/tests/badge.svg)
[![codecov](https://codecov.io/gh/prob-ml/tfmdoc/branch/master/graph/badge.svg?token=MD2RYN1AMA)](https://codecov.io/gh/prob-ml/tfmdoc)

#### Prayag Chatha & Jeffrey Regier

## Overview

This model is a transformer adapted to insurance claims data (specifically: Optum Claims), though the approach should generalize to EHRs. Each patient has a "history" of tokens (i.e. medical codes) that correspond to the diagnoses, lab results, medications, etc. they receive as they interact with the medical system. While individual patient records are "sparse" (only approximate the ground truth) and often messy, when we scale up to millions of patients, this constitutes a wealth of observational data that is hard for humans to interpret. We are working on the following problems:

### Prediction of ALD
Given sequences of patient medical data, the transformer learns to identify who will go on to get ALD (alcoholic liver disease) in the next month (or N number of days). Early prediciton of ALD is of interest because it is often lethal when it's eventually detected by humans. We can generalize this problem to more common chronic diseases, such as heart disease.

### Distinguishing Early and Late Stage ALD
We identify patients with ALD and transform them each into two records: an "late stage" record (data truncated 30 days before diagnosis) and an "early stage" record (data truncated 180 days before diagnosis, say). This is a harder prediction problem than distinguishing ALD from non-ALD subjects. Intuitively, the temporal spacing and ordering of tokens matters here, and this gives transformers an extra edge over non-sequential models. 

### Doubly-Robust Causal Inference: Medication Safety
While the transformer and other neural network models can be hard-to-interpret black boxes, they can still be useful for causal inference on observational data. Observational data tends to have a confounding structure of treatment and outcome, unlike randomized trials. We can use the transformer to model both *propensity* (e.g. likelihood of an individual to be prescribed a given drug) and *outcome* (e.g. the probability of a negative side effect) and combine these regression models to obtain an estimate of the true causal effect of treatment on outcome. Under the [doubly robust](https://www.law.berkeley.edu/files/AIPW(1).pdf) approach, if just one model is unbiased, the average treatment effect (ATE) estimator is unbiased. We draw inspiration here from the [Sentinel Initiative](https://www.sentinelinitiative.org/).


### Installation
```
poetry install
poetry shell
pre-commit install
```
