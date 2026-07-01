"""Pre-flight check: run this before the pipeline to catch issues early.

Usage:
    cd SerieAwithPandas/defr
    python preflight_check.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

DEFR_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEFR_DIR.parent
TEAM_TREND = REPO_ROOT / "team_trend"
MATCHES_CLASSIFICATION = REPO_ROOT / "matches_classification"

OK = "✓"
FAIL = "✗"
WARN = "⚠"

errors = []
warnings = []


def check(label: str, condition: bool, fix: str = ""):
    status = OK if condition else FAIL
    print(f"  {status}  {label}")
    if not condition:
        errors.append(f"{label}: {fix}")


def warn(label: str, condition: bool, fix: str = ""):
    status = OK if condition else WARN
    print(f"  {status}  {label}")
    if not condition:
        warnings.append(f"{label}: {fix}")


print("=" * 60)
print("DefR pipeline — pre-flight check")
print("=" * 60)

# ─── 1. Python version ────────────────────────────────────────────────
print("\n[1] Python version")
check(
    f"Python >= 3.10 (yours: {sys.version.split()[0]})",
    sys.version_info >= (3, 10),
    "Upgrade to Python 3.10+",
)

# ─── 2. Required packages ────────────────────────────────────────────
print("\n[2] Required packages")
packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
}
for import_name, pip_name in packages.items():
    try:
        importlib.import_module(import_name)
        ok = True
    except ImportError:
        ok = False
    check(f"{pip_name}", ok, f"pip install {pip_name}")

# ─── 3. Directory structure ───────────────────────────────────────────
print("\n[3] Directory structure")
check(
    f"defr/ exists at {DEFR_DIR}",
    DEFR_DIR.exists(),
    "You're not running from the right location",
)
check(
    f"defr_implementation/ package exists",
    (DEFR_DIR / "defr_implementation" / "__init__.py").exists(),
    "Missing defr_implementation package",
)
check(
    f"team_trend/ exists at {TEAM_TREND}",
    TEAM_TREND.exists(),
    "team_trend/ must be a sibling directory of defr/",
)

# ─── 4. matches_classification pipeline files ─────────────────────────
print("\n[4] matches_classification pipeline files")
required_files = [
    "pipeline.py",
    "config.py",
    "features.py",
    "backtesting.py",
    "aggregations.py",
    "cleaning.py",
]
for f in required_files:
    check(
        f"matches_classification/{f}",
        (MATCHES_CLASSIFICATION / f).exists(),
        f"Missing {f} in matches_classification/",
    )

check(
    "matches_classification/models/base.py",
    (MATCHES_CLASSIFICATION / "models" / "base.py").exists(),
    "Create matches_classification/models/ with __init__.py and base.py",
)

check(
    "matches_classification/models/logistic_pipeline.py",
    (MATCHES_CLASSIFICATION / "models" / "logistic_pipeline.py").exists(),
    "Missing logistic_pipeline.py in matches_classification/models/",
)

# ─── 5. FBref CSV ─────────────────────────────────────────────────────
print("\n[5] FBref CSV")
csv_path = MATCHES_CLASSIFICATION / "data" / "raw" / "matches_seriea.csv"
check(
    f"matches_seriea.csv at {csv_path.relative_to(REPO_ROOT)}",
    csv_path.exists(),
    f"CSV not found at {csv_path}. "
    "Edit FBREF_CSV in defr_implementation/config.py and inject_defr.py "
    "to point to the correct location.",
)
if not csv_path.exists():
    # Try to find it elsewhere
    for search_dir in [TEAM_TREND, MATCHES_CLASSIFICATION]:
        candidates = list(search_dir.rglob("matches_seriea.csv"))
        if candidates:
            print(f"       Found at: {candidates[0].relative_to(REPO_ROOT)}")
            break

# ─── 6. Output directories ───────────────────────────────────────────
print("\n[6] Output directories (will be created if missing)")
for subdir in ["output/plots", "output/data", "output/injection",
               "output/validation/plots", "data/wyscout_matches"]:
    p = DEFR_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    check(f"{subdir}/", p.exists(), "")

# ─── 7. Known pipeline bug ───────────────────────────────────────────
print("\n[7] Known pipeline bug check")
pipeline_path = MATCHES_CLASSIFICATION / "pipeline.py"
if pipeline_path.exists():
    content = pipeline_path.read_text()
    has_reassignment = "raw_df = add_match_features" in content
    has_bare_call = (
        "add_match_features(raw_df)" in content
        and "raw_df = add_match_features" not in content
    )
    if has_bare_call:
        print(f"  {WARN}  pipeline.py calls add_match_features(raw_df) without reassignment")
        print(f"       FIX: change to raw_df = add_match_features(raw_df)")
        print(f"       This bug causes rolling features (last_5_*, weighted_form)")
        print(f"       to be lost due to an in-function sort_values rebinding.")
        warnings.append("pipeline.py: add_match_features needs reassignment")
    else:
        check("pipeline.py: add_match_features uses reassignment", has_reassignment, "")

# ─── 8. Internet access (for Wyscout download) ────────────────────────
print("\n[8] Internet access (for Phase 1 Wyscout download)")
try:
    import urllib.request
    urllib.request.urlopen("https://github.com", timeout=5)
    check("Can reach github.com", True, "")
except Exception:
    warn(
        "Cannot reach github.com",
        False,
        "Phase 1 needs internet to download Wyscout data. "
        "Skip if data/wyscout_matches/ already has 380 JSON files.",
    )

existing_jsons = list((DEFR_DIR / "data" / "wyscout_matches").glob("*.json"))
if existing_jsons:
    check(f"Wyscout data already cached ({len(existing_jsons)} files)", len(existing_jsons) >= 380, "")

# ─── 9. Rescue path artifacts (optional) ─────────────────────────────
print("\n[9] Rescue path artifacts (optional — for full-bridge analysis)")
passing_parquet = DEFR_DIR / "output" / "injection" / "fbref_passing_data.parquet"
full_parquet = DEFR_DIR / "output" / "injection" / "fbref_with_defr_full.parquet"

if passing_parquet.exists():
    check("fbref_passing_data.parquet present (from fetch_passing_data.py)", True, "")
else:
    print(f"  -  fbref_passing_data.parquet not present.")
    print(f"     Run `python fetch_passing_data.py` LOCALLY to fetch")
    print(f"     n_opp_passes from FBref via soccerdata. This unlocks")
    print(f"     the full 10-feature bridge (R² = 0.59).")

if full_parquet.exists():
    check("fbref_with_defr_full.parquet present (from inject_defr_full.py)", True, "")
else:
    print(f"  -  fbref_with_defr_full.parquet not yet built. Run inject_defr_full.py")
    print(f"     after fetch_passing_data.py to apply the full bridge.")

# ─── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"ERRORS ({len(errors)}) — fix these before running:\n")
    for e in errors:
        print(f"  {FAIL}  {e}")
if warnings:
    print(f"\nWARNINGS ({len(warnings)}) — may cause issues:\n")
    for w in warnings:
        print(f"  {WARN}  {w}")
if not errors and not warnings:
    print("ALL CHECKS PASSED — ready to run the pipeline.")
    print()
    print("Run order:")
    print("  1. python run_defr_analysis.py        # Phase 1: Wyscout → bridge")
    print("  2. python refit_bridge_fbref.py        # Phase 2a: refit for FBref")
    print("  3. python inject_defr.py               # Phase 2b: apply reduced bridge")
    print("  4. python walkforward_validate.py      # Phase 2c: walk-forward test")
    print("  5. python make_validation_plots.py     # Phase 2d: plots")
    print("  6. python make_validation_report.py    # Phase 2e: HTML report")
    print()
    print("  Optional rescue path (full bridge with defensive features):")
    print("  R1. python sanity_check_rescue.py     # confirm rescue worth pursuing")
    print("  R2. python refit_bridge_rescue.py     # rebuild bridge with defensive features")
    print("  R3. python fetch_defensive_data.py    # LOCAL ONLY — needs internet + browser")
    print("  R4. python inject_defr_full.py        # apply rescue bridge")
    print("  R5. python walkforward_validate_full.py  # validate vs reduced + baseline")
    print()
    print("  Phase 4 — Team profiling and tactical clustering:")
    print("  P1. python fetch_defense_season_data.py  # OPTIONAL local fetch")
    print("  P2. python build_profiles_and_cluster.py # team-season profiles + K-means")
    print("  P3. python make_profile_report.py        # plots + HTML report")
elif not errors:
    print("\nNo blocking errors. Pipeline may run but check the warnings.")
print("=" * 60)
