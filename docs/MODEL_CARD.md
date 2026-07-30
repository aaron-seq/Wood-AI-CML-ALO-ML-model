# Model card: CML elimination classifier

## Overview

| | |
| --- | --- |
| Task | Binary classification — should this CML be retired from the monitoring programme? |
| Architecture | scikit-learn `Pipeline`: `StandardScaler` + `OneHotEncoder` → `RandomForestClassifier` |
| Hyperparameters | 300 trees, `max_depth=10`, `min_samples_leaf=2`, `class_weight="balanced_subsample"` |
| Selection | 5-fold grid search optimising F1 |
| Artifact | `models/cml_elimination_model.joblib`, pickled by scikit-learn 1.7.2 |
| Training data | `data/sample_cml_data.csv` — 200 synthetic rows, 21 positive (10.5%) |

## Inputs

Nine features. The first six are numeric and standardised; the last three are
one-hot encoded with `handle_unknown="ignore"`, so an unseen category
contributes nothing rather than raising.

| Feature | Source |
| --- | --- |
| `average_corrosion_rate` | Supplied |
| `thickness_mm` | Supplied |
| `corrosion_thickness_ratio` | Derived: rate ÷ thickness |
| `remaining_life_years` | Supplied if present, else `(t − 3.0) ÷ rate` |
| `risk_score` | Supplied if present, else derived from rate and thickness |
| `days_since_inspection` | Derived from `last_inspection_date`, 365 if unknown |
| `commodity`, `feature_type`, `cml_shape` | Supplied |

`remaining_life_years` and `risk_score` are used **as supplied** when present,
because the training pipeline consumed them from the source file. Recomputing
them at inference caused train/serve skew
([ADR-0003](adr/0003-share-feature-engineering-between-training-and-serving.md)).

## Output

`elimination_probability` is the **fraction of trees voting to eliminate**. It
is *not* a calibrated probability: 0.9 does not mean "90% likely correct." The
`HIGH`/`MODERATE` label is distance from the 0.5 boundary (threshold 0.3), not
a validated confidence interval. Calibration is
[planned work](DEPLOYMENT.md#future-work).

## Feature importances

Read from the served artifact:

| Feature | Importance |
| --- | --- |
| `corrosion_thickness_ratio` | 37.2% |
| `average_corrosion_rate` | 29.2% |
| `risk_score` | 9.7% |
| `remaining_life_years` | 7.1% |
| `thickness_mm` | 6.5% |
| `days_since_inspection` | 3.0% |
| commodity / feature type / shape | 7.3% combined |

Degradation rate relative to remaining wall dominates, which is at least
consistent with how a corrosion engineer reasons. Commodity and geometry
contribute little — plausibly because the synthetic generator did not encode
much commodity-specific behaviour, not because they do not matter in the field.

## Performance

**The served artifact's headline metrics are not a usable estimate.** Its
metadata reports accuracy 0.95 and ROC-AUC 0.99, computed on a 40-row test
split containing **four** positive examples. At that size a single
classification flips the metrics by tens of points. These were previously
published in the README as the model's performance; they should not have been.

Retraining the identical pipeline on the larger 500-row dataset:

| Metric | 200-row set (as shipped) | 500-row set (measured) |
| --- | --- | --- |
| Test rows / positives | 40 / 4 | 100 / 39 |
| Accuracy | 0.95 | 0.68 |
| Precision (eliminate) | — | 0.62 |
| Recall (eliminate) | — | 0.46 |
| F1 (eliminate) | 0.75 | 0.53 |
| ROC-AUC | 0.99 | 0.69 |
| CV F1 (mean ± sd) | 0.87 ± 0.12 | 0.61 ± 0.05 |
| Confusion matrix | `[[35,1],[1,3]]` | `[[50,11],[21,18]]` |

**Treat ROC-AUC ≈ 0.69 as the honest figure.** The larger evaluation has 39
positives — still small, but an order of magnitude more informative. On that
set the model misses more than half the true elimination candidates
(recall 0.46).

Reproduce:

```bash
make train DATASET=data/cml_sample_500.csv
```

The two datasets differ in class balance (10.5% vs 39.4% positive), so their
numbers are not directly comparable. That divergence is itself a signal that
neither is a settled ground truth.

## Intended use

**In scope.** Pre-screening a CML population to focus engineering review;
ranking candidates by elimination probability; quantifying how often experts
agree with the model via `/sme-override` statistics.

**Out of scope.** Autonomous retirement of monitoring locations. Any use where
a false negative — keeping a CML that should be eliminated — is not caught by
human review. Fitness-for-service or remaining-life determinations for
regulatory submission. Assets whose commodity, geometry or corrosion mechanism
is unlike the training distribution.

## Limitations

1. **Trained entirely on synthetic data.** Both bundled datasets are generated
   (`scripts/generate_500_row_dataset.py`). No metric here predicts field
   performance. Validating against real inspection outcomes is the single most
   important outstanding task.
2. **Small evaluation sets.** 40 and 100 test rows. Confidence intervals on
   every metric above are wide.
3. **Recall is low** (0.46 on the larger set) — the model misses more than half
   of the true elimination candidates. Because a missed elimination costs an
   unnecessary inspection rather than a safety event, this is the safer
   direction to err, but it caps the achievable savings.
4. **Uncalibrated probabilities.** See Output.
5. **No temporal validation.** The split is random, not chronological, so
   nothing tests generalisation to future inspection rounds.
6. **Minimum thickness is a single global constant** (3.0 mm). Real programmes
   set it per circuit from design pressure and material.
7. **No drift monitoring.** Nothing detects the input distribution moving away
   from training.

## Ethical and safety considerations

This model informs decisions about pressure-equipment inspection, where an
error can eventually contribute to a loss of containment. Two safeguards are
structural rather than advisory:

- **Human-in-the-loop by design.** The SME override system records the
  engineer's decision, their reasoning and the model's original prediction —
  an audit trail, not a rubber stamp.
- **Errors are asymmetric.** A false positive (eliminate a CML that should be
  kept) removes a monitoring point and is the consequential mistake. Class
  weighting and F1 selection lean against it, but the model's `precision` of
  0.62 means roughly one in three eliminations is wrong on the larger dataset.
  Expert review of every proposed elimination is not optional.

## Maintenance

| | |
| --- | --- |
| Retrain when | New labelled inspection outcomes arrive; input distribution shifts; scikit-learn pin is raised |
| Version pinning | `scikit-learn>=1.7.2,<1.8` — the artifact is a pickle; CI fails if it does not load warning-free |
| Registry | Timestamped artifacts and metadata JSONs accumulate in `models/`; the latest is copied to `cml_elimination_model.joblib` |
| Rollback | Copy a previous timestamped artifact over the canonical filename and restart — see [DEPLOYMENT.md](DEPLOYMENT.md#rollback) |
