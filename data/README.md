# Data

Both datasets are **synthetic**. No row describes a real asset, and no
metric computed on them predicts field performance. They exist to exercise
the pipeline and to show the expected schema.

| File | Rows | `elimination_flag = 1` | Provenance |
| --- | --- | --- | --- |
| `cml_sample_500.csv` | 500 | 197 (39.4%) | `scripts/generate_500_row_dataset.py` (rules below) |
| `sample_cml_data.csv` | 200 | 21 (10.5%) | Committed in the initial skeleton; no generator survives |

The served model (`models/cml_elimination_model.joblib`) was trained on the
**200-row** file. Its metadata does not name the dataset, but its test split —
40 rows, 4 positive — is exactly the 20% stratified split the trainer takes of
that file; the 500-row file would give 100. The honest metrics in the
[model card](../docs/MODEL_CARD.md) come from retraining on the **500-row** file.

> **Do not concatenate the two files.** They reuse the same identifiers
> (`CML-001` … `CML-200`) for unrelated records, and 82 of those 200 shared
> ids carry contradictory labels. Joining them trains on noise.

The SME override audit trail (`sme_overrides.json`) is written here at
runtime. It is per-deployment state and deliberately not tracked: a fresh
install starts with no overrides.

## Columns

The first six are required by every endpoint; see the
[README](../README.md#input-data) for what happens when the optional ones are
absent.

| Column | Type | Unit / values | Meaning |
| --- | --- | --- | --- |
| `id_number` | string | e.g. `CML-042` | CML identifier; unique within a file |
| `average_corrosion_rate` | float | mm/year | Measured wall loss rate |
| `thickness_mm` | float | mm | Current measured wall thickness |
| `commodity` | string | Condensate, Crude Oil, Fuel Gas, Gas, Natural Gas, Potable Water, Process Water, Refined Product, Steam | Process fluid |
| `feature_type` | string | Bend, Elbow, Flange, Header, Nozzle, Pipe, Reducer, Tee, Weld | Component monitored |
| `cml_shape` | string | Both, External, Internal | Where the measurement is taken |
| `last_inspection_date` | date | `YYYY-MM-DD` | Most recent inspection |
| `next_inspection_date` | date | `YYYY-MM-DD` | Scheduled next inspection. Not a model input |
| `remaining_life_years` | float | years | API 570 `(t − 3.0 mm) / rate`, capped at 300 by the generator (the API caps derived values at 50) |
| `days_to_next_inspection` | int | days | `next_inspection_date` minus the **generation** date — stale the day after, and negative for most rows now. Not a model input |
| `risk_score` | int | 0–100 | Rule-based score, below |
| `elimination_flag` | int | 0 / 1 | **Target.** 1 = the CML could be retired from the monitoring programme |

Values outside the documented ranges are scored, not rejected; the API
reports them as `warnings`.

## How the 500-row labels were made

`scripts/generate_500_row_dataset.py` draws corrosion rates by commodity and
thickness by feature type from uniform ranges, then labels each row with the
first matching rule:

| Rule | `elimination_flag = 1` with probability |
| --- | --- |
| rate < 0.08 and thickness > 9.5 | 0.7 |
| rate < 0.05 | 0.8 |
| `risk_score` > 70 | 0 |
| `risk_score` > 50 | 0.2 |
| otherwise | 0.3 |

`risk_score` adds 40 / 25 / 10 for a rate above 0.15 / 0.10 / otherwise, 30 / 15
for thickness below 8.0 / 9.5 mm, and 30 / 15 for remaining life below 20 / 50
years, capped at 100.

Two consequences matter when reading any metric:

- **The labels carry deliberate randomness.** Every rule but one is a coin
  flip with a bias, so a perfect model of the rules still misclassifies a
  large share of rows. The ROC-AUC of about 0.69 in the model card measures
  how much signal survives that noise, not how good the model could be on
  real outcomes.
- **The rules are not engineering judgement.** They were written to produce
  plausible-looking data. Training on them teaches the model the generator,
  not inspection practice.

The script seeds its random generators (42), but dates are computed from the
clock at generation time, so re-running it changes every date-derived column.

## Real data

Replacing these files with real inspection outcomes is the prerequisite for
any production use. Keep the column names above, label `elimination_flag`
from actual retirement decisions, and record where the data came from here.
