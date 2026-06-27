# DefR — Defensive Responsibility proxy pipeline

A reproducible end-to-end analysis that:

1. Builds a team-level **DefR metric** on Wyscout Serie A 2017/18
   event data (380 matches, ~647k events)
2. Fits a **bridge regression** mapping FBref aggregate stats to DefR
3. **Injects** the resulting proxy into the seriedAta FBref data (2020–2025)
4. Runs **walk-forward validation** to test whether the proxy adds
   predictive value to the existing `team_trend` features
5. Documents the result in two **self-contained HTML reports**

The result is an honest negative finding: the FBref-compatible proxy
carries real defensive signal but is **redundant** with the existing
strength and form features in the pipeline. See
`output/defr_injection_report.html` for the full walk-forward analysis.

## Project structure

```
SerieAwithPandas/
├── defr/                                    # DefR project root
│   ├── defr_implementation/                 # Python package
│   │   ├── __init__.py
│   │   ├── config.py                        # paths, zone grid, event taxonomy
│   │   ├── data.py                          # Wyscout download + JSON parsing
│   │   ├── model.py                         # zones, classification, DefR scoring
│   │   ├── bridge.py                        # aggregate features + Ridge bridge
│   │   ├── plots.py                         # matplotlib visualizations
│   │   └── report.py                        # standalone HTML report builder
│   ├── output/                              # all generated outputs
│   │   ├── defr_analysis_report.html        # Phase 1 report
│   │   ├── defr_injection_report.html       # Phase 2 report
│   │   ├── plots/                           # Phase 1 plots (6 PNGs)
│   │   ├── data/                            # Phase 1 artifacts (parquet + JSON)
│   │   ├── injection/                       # Phase 2 artifacts
│   │   │   ├── bridge_regression_fbref.json # reduced bridge coefficients
│   │   │   └── fbref_with_defr.parquet      # FBref + defr_proxy columns
│   │   └── validation/                      # Phase 2 walk-forward results
│   │       ├── plots/                       # 5 validation PNGs
│   │       ├── all_folds.csv
│   │       ├── summary.csv
│   │       ├── f1_pivot.csv
│   │       └── paired_tests.json
│   ├── data/wyscout_matches/                # downloaded JSONs (auto-fetched)
│   ├── __init__.py
│   ├── run_defr_analysis.py                 # Phase 1 orchestrator
│   ├── refit_bridge_fbref.py                # Phase 2: refit bridge
│   ├── inject_defr.py                       # Phase 2: apply bridge to FBref
│   ├── walkforward_validate.py              # Phase 2: walk-forward validation
│   ├── make_validation_plots.py             # Phase 2: validation plots
│   ├── make_validation_report.py            # Phase 2: validation HTML report
│   ├── README.md
│   └── requirements.txt
├── matches_classification/                   # W/D/L classification pipeline
│   ├── data/raw/matches_seriea.csv
│   ├── models/
│   │   ├── base.py
│   │   └── logistic_pipeline.py
│   ├── pipeline.py
│   ├── config.py
│   ├── features.py
│   ├── backtesting.py
│   └── ...
└── ...
```

All paths are computed relative to each script's own location. No
hardcoded absolute paths. The `defr_implementation/config.py` module
resolves `REPO_ROOT` as `Path(__file__).parent.parent.parent` and
derives all other paths from that.

## Headline results

### Phase 1 — Wyscout bridge (`run_defr_analysis.py`)

- 647,372 events parsed across 380 matches, 20 teams
- Zone baseline rates range from 0.12 (high in the field) to 1.06
  (own-goal line) across a 6×4 grid
- Season DefR rankings match known team identities: Atalanta (+18.0)
  and Fiorentina (+28.2) high; Napoli (−24.2) low
- Bridge regression: CV R² = **0.59**, MAE ≈ 19 (10 features)

### Phase 2 — FBref injection

- Reduced bridge (FBref-only, 7 features): CV R² = **0.14** (Δ = −0.45)
- Walk-forward F1 macro on 4 seasonal folds:
  - baseline (23 features): **0.4919**
  - +rolling proxy: **0.4919** (exact zero delta)
  - +match proxy: **0.4882** (−0.0037, hurts)
- The proxy correlates +0.58 with `cum_avg_points` — redundant, not null

## Reproducing the analysis

### Prerequisites

Python 3.10+. From the `defr/` directory:

```bash
pip install -r requirements.txt
```

### Run Phase 1 (Wyscout → bridge)

```bash
cd SerieAwithPandas/defr
python run_defr_analysis.py
```

First run downloads ~50 MB of Wyscout match JSONs (~2–3 min);
subsequent runs reuse the cache in `data/wyscout_matches/`.

### Run Phase 2 (injection + walk-forward)

```bash
cd SerieAwithPandas/defr

# Step 1: Refit bridge with FBref-only features
python refit_bridge_fbref.py

# Step 2: Inject defr_proxy into FBref data
python inject_defr.py

# Step 3: Walk-forward validation (requires team_trend pipeline)
python walkforward_validate.py

# Step 4: Generate plots and report
python make_validation_plots.py
python make_validation_report.py
```

Phase 2 expects `matches_classification/` to be a sibling directory of
`defr/` containing the classification pipeline files (`pipeline.py`,
`config.py`, `features.py`, `backtesting.py`, `models/base.py`,
`models/logistic_pipeline.py`) and the FBref CSV at
`matches_classification/data/raw/matches_seriea.csv`.

**Note:** there is a known bug in `matches_classification/pipeline.py` where
`add_match_features(raw_df)` is called without reassignment. The function
internally does `df = df.sort_values(...)` which rebinds the local
variable, losing rolling features. The fix: `raw_df = add_match_features(raw_df)`.

## Key transferable artifacts

**`output/injection/bridge_regression_fbref.json`** — the fitted Ridge
coefficients and scaler parameters. Apply to any FBref-formatted data:

```python
import json, numpy as np

with open("output/injection/bridge_regression_fbref.json") as f:
    bridge = json.load(f)

features = bridge["feature_names"]
means = np.array([bridge["scaler_means"][f] for f in features])
stds  = np.array([bridge["scaler_stds"][f]  for f in features])
coefs = np.array([bridge["coefficients"][f] for f in features])

x_std = (x_new - means) / stds        # x_new shape: (n, 7)
defr_proxy = x_std @ coefs + bridge["intercept"]
```

## Conclusions

1. **Do not inject `defr_proxy` into `FEATURES_CLEAN`** in its current
   form. The walk-forward F1 delta is exactly zero across all four folds.
2. **The work is reliable** — the negative finding is robust. Three
   independent mechanisms (R² collapse, feature correlations, team OHE
   absorption) all predict no marginal value.
3. **The most promising revival path** is obtaining pass-volume data for
   2020–2025 (the dominant bridge feature). That would restore CV R² to
   ≈ 0.59 and produce a proxy capturing defensive style rather than
   team quality.

## Data sources

- **Wyscout Open Data**: Pappalardo et al. (2019), Scientific Data 6:236.
  Mirror: `github.com/koenvo/wyscout-soccer-match-event-dataset`
- **FBref Serie A 2020–2025**: `matches_seriea.csv` via `soccerdata`
- **Conceptual reference**: Hudl/StatsBomb, *Defensive Responsibility:
  A New Way To Measure Defensive Output*
