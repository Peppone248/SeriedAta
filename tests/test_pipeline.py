from pipeline import run_pipeline


def test_pipeline_runs():
    outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
    assert "raw_df" in outputs
    assert "team_stats" in outputs
    assert "team_season_stats" in outputs


def test_result_values_are_valid():
    outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
    summary = outputs["validation_summary"]
    assert summary["valid_result_values"] is True


def test_venue_values_are_valid():
    outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
    summary = outputs["validation_summary"]
    assert summary["valid_venue_values"] is True


def test_matches_balance():
    outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
    checks = outputs["aggregation_checks"]
    assert checks["matches_balance_ok"] is True


def test_one_champion_per_season():
    outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
    champions = outputs["season_champions"]
    seasons = outputs["team_season_stats"]["season"].nunique()
    assert len(champions) == seasons


def test_home_away_merge_is_one_row_per_team():
    outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
    venue_merged = outputs["venue_merged"]
    assert venue_merged["team"].nunique() == len(venue_merged)

def test_points_values():
        outputs = run_pipeline("data/raw/matches_seriea.csv", save=False)
        raw_df = outputs["raw_df"]
        assert set(raw_df["points"].unique()).issubset({0, 1, 3})