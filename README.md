# Serie dAta

Project based on Serie A match statistics (2020–2025). Has the aim to predict matches' outcome based on statistics
creating aggregated features based on the datasets. Try to understand which features influence most the predictions without considering match events and players individual statistics.
It covers the full pipeline from raw CSV ingestion through feature engineering, multi-model classification, regression, and per-match interpretability reports.

---

## Project structure

```text
seriedAta/
├── README.md
│
├── data/
│   ├── raw/
│   │   └── matches_seriea.csv
│   └── processed/
│
├── models/
│   ├── base.py                    # ClassificationResult / RegressionResult dataclasses
│   ├── logistic_pipeline.py       # Logistic Regression pipeline
│   ├── xgboost_pipeline.py        # XGBoost pipeline (temporal split)
│   └── lgbm_pipeline.py           # LightGBM pipeline (temporal split)
│
├── dashboard_teams/
│   └── dashboard.py               # Streamlit team dashboard
│
├── reports/
│   └── summary.md
│
├── main.py                        # Single entry point
├── config.py                      # Feature lists and constants (single source of truth)
├── pipeline.py                    # ETL orchestrator
├── features.py                    # Feature engineering
├── cleaning.py                    # Schema cleanup and dtype conversion
├── aggregations.py                # Grouped analysis tables
├── stats.py                       # Descriptive statistics
├── validation.py                  # Data quality checks
├── evaluation.py                  # Unified model comparison (leaderboard + plots)
├── model_comparison.py            # Multi-model comparison on identical temporal split
├── classification_model_interpretation.py          # Logistic feature importance
├── classification_model_interpretation_xgboost.py  # XGBoost SHAP explanations
├── linear_regression_model.py     # Goal-diff regression
├── output_utils.py
├── reporting.py
└── __init__.py
```

---

## What each file does

### Data layer

| File | Responsibility |
|---|---|
| `pipeline.py` | Loads the CSV, runs cleaning, validation and feature engineering, returns all output tables in a dict |
| `cleaning.py` | Schema normalisation, dtype conversion, column renaming |
| `features.py` | All engineered football features: form windows, cumulative strength, head-to-head rates, fatigue |
| `aggregations.py` | Grouped analysis tables: `team_stats`, `team_season_stats`, `home_away_stats` |
| `stats.py` | Descriptive statistics outputs |
| `validation.py` | Data quality and consistency checks |

### Configuration

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for all feature lists (`LOGISTIC_NUM_FEATURES`, `XGBOOST_NUM_FEATURES`, `LGBM_NUM_FEATURES`, `CAT_FEATURES`, `REGRESSION_FEATURES`). All pipelines import from here. |

### ML models

| File | Responsibility |
|---|---|
| `models/base.py` | `ClassificationResult` and `RegressionResult` dataclasses — unified output contract so all models are directly comparable |
| `models/logistic_pipeline.py` | Logistic Regression with `GridSearchCV` on `TimeSeriesSplit`, returns `ClassificationResult` |
| `models/xgboost_pipeline.py` | XGBoost with temporal train/test split and `LabelEncoder`, returns `ClassificationResult` |
| `models/lgbm_pipeline.py` | LightGBM with temporal split (no `LabelEncoder` needed), returns `ClassificationResult` |
| `linear_regression_model.py` | Linear regression baseline for goal-diff prediction, returns a dict wrappable into `RegressionResult` |

### Evaluation and comparison

| File | Responsibility |
|---|---|
| `evaluation.py` | `classification_leaderboard()`, `plot_classification_comparison()`, `plot_confusion_matrices()`, `regression_leaderboard()` — all operate on lists of result dataclasses, no model-specific logic |
| `model_comparison.py` | Trains LR Baseline, LR Tuned, Random Forest, XGBoost and LightGBM on the **same** temporal split for a fair head-to-head comparison |

### Interpretability

| File | Responsibility |
|---|---|
| `classification_model_interpretation.py` | Per-class coefficient plots and probability distributions for Logistic Regression |
| `classification_model_interpretation_xgboost.py` | SHAP-based global importance, per-class importance, beeswarm plots, and single-match waterfall explanation for XGBoost |

### Entry points

| File | Responsibility |
|---|---|
| `main.py` | Orchestrates the full run: data pipeline → classifiers → regression → model comparison → interpretability |
| `dashboard_teams/dashboard.py` | Streamlit dashboard for interactive team-level exploration |
| `reporting.py` | Generates `reports/summary.md` |

---

## Run

```bash
python main.py
```

To run only the multi-model comparison (LR, RF, XGBoost, LightGBM on identical split):

```python
from model_comparison import run_model_comparison, plot_comparison_results

results = run_model_comparison(df)
plot_comparison_results(results["leaderboard"])
```

To run a single pipeline in isolation:

```python
from models.xgboost_pipeline import run_classification_pipeline
from evaluation import print_classification_leaderboard

result = run_classification_pipeline(df)
print_classification_leaderboard([result])
```

---

## ML architecture

All three classifier pipelines share an identical output contract (`ClassificationResult`) so results can be compared without model-specific code:

```
ClassificationResult
  ├── accuracy, f1_macro, f1_per_class {"L", "D", "W"}
  ├── log_loss
  ├── predictions, probabilities
  ├── prediction_table, probability_table
  └── model (fitted GridSearchCV)
```

The `evaluation.py` module works on `list[ClassificationResult]` — adding a new model means implementing the pipeline, returning a `ClassificationResult`, and appending it to the list.

Feature sets for each model are defined in `config.py`:

- **Logistic** uses `LOGISTIC_NUM_FEATURES` (includes `finishing_efficiency`, `defensive_efficiency`)
- **XGBoost and LightGBM** use their respective extended sets (adds `dist`, `h2h_win_rate`, `form_consistency`)
- **`model_comparison.py`** uses `NUM_FEATURES` (alias for the logistic set) so all models train on identical features for a fair comparison

## ML models comparision
<img width="1800" height="500" alt="models_comparison" src="https://github.com/user-attachments/assets/4cc06804-9c3e-48e5-b184-e6bb5980e146" />

## SHAP features interpretability
<img width="1000" height="600" alt="top20_features_shap" src="https://github.com/user-attachments/assets/59353ca9-cae2-425f-baf6-b5266e320b92" />

---

## Design notes

- `raw_df = pd.read_csv(...)` → `df = raw_df.copy()` pattern throughout: raw data is never mutated.
- Temporal splits (`temporal_train_test_split`) are used in tree-based models to prevent leakage from future matches into training.
- Cumulative strength features use `shift(1).expanding().mean()` so each row only sees past matches.
- Division-by-zero in ratio features is guarded with `np.where(denominator > 0, num / denom, np.nan)`.
- SHAP explanations live entirely in `classification_model_interpretation_xgboost.py` and are decoupled from the training pipeline.
