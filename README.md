# Serie dAta ⚽

A data science project built around **Serie A football** (seasons 2020–2025). The goal is to explore how far you can go in understanding and predicting match outcomes using only publicly available team-level and squad-level statistics — no match events, no live tracking, no betting odds.

The project grew organically across three areas, each asking a different question about the same sport.

---

## The three pillars

### 🎯 Match outcome classification — `matches_classification/`

*Can we predict whether a match ends in a Win, Draw, or Loss using pre-match aggregate features?*

This is where the project started. The pipeline takes raw FBref match data and engineers dozens of features capturing team form, cumulative strength, head-to-head history, and tactical parity. Three classification models (Logistic Regression, XGBoost, LightGBM) plus a cascaded two-stage classifier are trained and compared through walk-forward backtesting across seasons — no future data ever leaks into training.

The best models reach ~0.50 F1 macro, which is roughly where the football prediction literature says aggregate-stats-only models top out. Draws are the hardest to predict (they tend to happen between evenly matched teams, exactly where the model's probability mass is most diffuse), while Win and Loss predictions benefit from clearer xG imbalances and strength differentials.

SHAP explanations make it possible to open up the XGBoost black box and understand *why* the model predicted a specific outcome for any given match.

### 📈 Squad momentum and season simulation — `team_trend/`

*Can we model how a team's form evolves across a season and simulate where they'll finish?*

This is the data engineering backbone. A full **Bronze → Silver → Gold** medallion ETL pipeline scrapes FBref player-level data via the `soccerdata` library, cleans and standardises player statistics, then aggregates them into squad-level momentum features (rolling averages, trend indicators, volatility measures).

On top of that sits an XGBoost regression model with quantile uncertainty bands, and a **Monte Carlo season simulation** that propagates match-by-match uncertainty into final standings distributions. On the 2024/25 season, the simulation achieved MAE 2.24 on final season points with 84.2% prediction interval coverage.

### 🛡️ Defensive Responsibility (DefR) — `defr/` · branch `DefR_workflow`

*Can we measure how aggressively a team defends relative to what the game demands, and does that information help predictions?*

> **This work lives on the `DefR_workflow` branch.**

Inspired by the Hudl/StatsBomb article *"Defensive Responsibility: A New Way To Measure Defensive Output"*, this module builds a team-level DefR metric from ~647,000 Wyscout event records (Serie A 2017/18). The idea: for every opposition attacking action, compute how many defensive responses the zone-level baseline predicts, then measure each team's gap between actual and expected defensive output.

The metric produces footballistically coherent results — Gasperini's aggressive Atalanta ranks high, Sarri's possession-dominant Napoli ranks low — and a bridge regression (R² = 0.59) quantifies how much of the spatial defensive signal survives when compressed into the aggregate statistics that FBref provides.

Two self-contained HTML reports document the full methodology, the results, and the limitations. The analysis is fully reproducible from source via `python run_defr_analysis.py`.

---

## Project structure

```text
SerieAwithPandas/
│
├── matches_classification/          # W/D/L classification pipeline
│   ├── data/raw/matches_seriea.csv
│   ├── models/
│   │   ├── base.py                  # ClassificationResult / RegressionResult contracts
│   │   ├── logistic_pipeline.py
│   │   ├── xgboost_pipeline.py
│   │   ├── lgbm_pipeline.py
│   │   └── cascaded_classification.py
│   ├── pipeline.py                  # ETL orchestrator
│   ├── config.py                    # Feature lists (single source of truth)
│   ├── features.py                  # All engineered features
│   ├── backtesting.py               # Walk-forward seasonal validation
│   ├── evaluation.py                # Leaderboard + comparison plots
│   ├── main.py                      # Full run entry point
│   └── ...
│
├── team_trend/                      # Squad momentum + season simulation
│   ├── data/
│   ├── etl/                         # Bronze → Silver → Gold layers
│   ├── scrapers/                    # FBref scraping via soccerdata
│   ├── models/
│   ├── visualization/
│   ├── pipeline.py                  # Medallion ETL orchestrator
│   ├── simulate_season.py           # Monte Carlo season simulation
│   ├── squad_features.py            # Player → squad aggregation
│   └── ...
│
├── defr/                            # Defensive Responsibility (branch: DefR_workflow)
│   ├── defr_implementation/         # Python package
│   │   ├── config.py, data.py, model.py, bridge.py, plots.py, report.py
│   ├── output/                      # Reports, plots, artifacts
│   ├── run_defr_analysis.py         # Phase 1: Wyscout → DefR → bridge
│   ├── README.md
│   └── ...
│
├── reports/
├── downloaded_files/
└── README.md                        # ← you are here
```

---

## Quick start

### Match classification

```bash
cd matches_classification
python main.py
```

This runs the full pipeline: data loading → feature engineering → all classifiers → regression → model comparison → SHAP interpretability. Each section is a standalone function — comment out any you don't need.

### Team trend / season simulation

```bash
cd team_trend
python pipeline.py --start-from bronze --force
python simulate_season.py
```

Requires internet access for the initial FBref scrape. Subsequent runs use cached parquet files.

### DefR analysis

```bash
git checkout DefR_workflow
cd defr
python preflight_check.py    # verify dependencies
python run_defr_analysis.py   # full pipeline
```

First run downloads ~50 MB of Wyscout data; subsequent runs reuse the cache.

---

## ML architecture

All classifiers share the same output contract (`ClassificationResult`), so adding a new model means implementing a pipeline that returns one — no changes needed elsewhere.

```
ClassificationResult
  ├── accuracy, f1_macro, f1_per_class {L, D, W}
  ├── log_loss
  ├── predictions, probabilities
  ├── prediction_table, probability_table
  └── model (fitted estimator)
```

Feature sets are defined in `matches_classification/config.py` — the single source of truth. Each model imports its feature list from there.

### Models comparison
<img width="1800" height="500" alt="models_comparison" src="https://github.com/user-attachments/assets/4cc06804-9c3e-48e5-b184-e6bb5980e146" />

### SHAP feature interpretability
<img width="1000" height="600" alt="top20_features_shap" src="https://github.com/user-attachments/assets/59353ca9-cae2-425f-baf6-b5266e320b92" />

---

## Design principles

A few choices that run through the whole project:

- **No future leakage.** Cumulative features use `shift(1).expanding().mean()` so each row only sees past matches. Walk-forward backtesting validates on seasons strictly after training. Standardisation is always fit on the training fold only.

- **Theory before code.** Major modelling decisions (feature groups, model families, the DefR bridge approach) are grounded in explicit hypotheses tested through ablation and walk-forward validation. Negative results are documented honestly.

- **Footballistic validation.** Statistical outputs are always sanity-checked against known football realities (Gasperini's pressing, Sarri's possession dominance, Bologna's late-season collapse) to catch metrics that look good on paper but don't make football sense.

- **Reproducibility.** Fixed random seeds, deterministic pipelines, self-contained HTML reports with embedded plots. Every experiment can be re-run from source.

---

## Data sources

- **FBref** (via `soccerdata` library) — match-level and player-level statistics for Serie A 2020–2025
- **Wyscout Open Data** — Pappalardo et al. (2019), *Scientific Data* 6:236 — event-level data for Serie A 2017/18 (used in the DefR analysis)