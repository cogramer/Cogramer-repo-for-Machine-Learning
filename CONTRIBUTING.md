# Contributing Guidelines

Thanks for your interest in contributing to this project!

---

# Project Scope

This project focuses on:

* Forecasting from multi-variable weather time series data
* Optimizing model training

Current model branches:

* `LSTMv0.0.9-beta`
* `LSTMv0.0.9-alpha`

---

# Ways to Contribute

You can contribute in several ways:

## Model Improvements

Examples:

* Better LSTM architectures
* Bidirectional LSTMs
* GRUs
* Transformer-based forecasting
* Attention mechanisms

---

## Preprocessing Improvements

Examples:

* Better missing timestep recovery
* Alternative interpolation strategies
* More physically-aware feature reconstruction
* Improved time encoding
* Better outlier handling

---

## Evaluation Improvements

Examples:

* Additional error metrics
* Visualization improvements
* Confidence interval estimation
* Seasonal performance analysis

---

## Dataset Expansion

Examples:

* Support for new locations
* Multi-city datasets
* Multi-climate testing

---

# Contribution Workflow

## 1. Fork the repository

Create your own fork.

---

## 2. Create a branch

Use descriptive names:

```bash

feature/improve-preprocessing

feature/add-transformer-model

fix/delta-gap-scaling

```

Avoid committing directly to `main` (please).

---

## 3. Make your changes

Keep changes focused.

Good:

* one feature
* one fix
* one experiment

Avoid mixing unrelated changes.

---

## 4. Test your changes

Before submitting:

* Verify training runs successfully
* Verify preprocessing shapes are correct
* Verify no NaN leaks into tensors
* Verify model evaluation completes

Minimum checks:

* `x\_train.shape`
* `x\_test.shape`
* training convergence
* inference output
* inverse transform correctness

---

## 5. Submit a Pull Request

PRs should clearly explain:

* what changed
* why it changed
* expected impact
* any tradeoffs

Good PR example:

> Added gap-aware humidity interpolation to reduce discontinuities in sparse datasets.

Bad PR example:

> fixed stuff

---

# Coding Standards

## General

* Keep code readable
* Prefer explicit logic over clever shortcuts
* Use descriptive variable names
* Keep preprocessing deterministic

I know my code isn't the best but I'll try to keep up to the standard.

---

## Comments

Comments should explain \*why\* some changes were made.

Good:

```python

# Solar radiance is forced to zero at night to avoid physically impossible interpolation


```

Bad:

```python

# Set solar radiance


```

---

## Formatting

Follow:

* PEP 8
* consistent spacing
* logical section headers

---

# Preprocessing Rules

If modifying preprocessing:

Do not:

* leak test data into training scaler
* derive cyclic features from row position unless intentional
* interpolate blindly across large gaps

Preferred:

* timestamp-based feature derivation
* gap-aware reconstruction
* physically constrained interpolation

---

# Model Rules

If modifying models:

Keep experiments reproducible.

Document:

* hidden size
* layer count
* dropout
* optimizer
* scheduler changes
* loss function changes

---

# Reporting Results

When submitting model changes, include:

## Dataset used

Example:

```text

24_09_18_to_26_06_24


```

---

## Metrics

Required:

* Test Loss
* MAPE
* MAE

Optional:

* RMSE
* Max Error

---

## Training graph

Include loss curves when possible.

---

# What Not to Submit

Please avoid:

* Random refactors without purpose
* Unverified performance claims
* Large unrelated dependency additions
* Hardcoded local file paths
* Raw API keys
* Database dumps unless requested

---

# Experimental Nature

This repository is still being developed.

Contributors should expect:

* architectural changes
* code rewrites
* metric shifts between versions

---

# Questions

If unsure whether a contribution fits the project, open an issue first.

Discussion before implementation is encouraged for larger changes.



