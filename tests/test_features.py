import pandas as pd
from src.features import build_features


def test_first_match_uses_prior_defaults_not_current_result():
    matches = pd.DataFrame([
        {"date": "2024-08-10", "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 0, "result": "H"},
        {"date": "2024-08-17", "home_team": "A", "away_team": "C",
         "home_goals": 0, "away_goals": 1, "result": "A"},
    ])
    result = build_features(matches)
    # The first A feature is the cold-start prior, not the three goals scored in match 1.
    assert result.loc[0, "home_goals_for_avg"] == 1.2
    # The second A feature can only see the first match.
    assert result.loc[1, "home_goals_for_avg"] == 3.0
    assert result.loc[1, "home_points_avg"] == 3.0


def test_feature_target_is_present_and_no_score_columns_are_predictors():
    matches = pd.DataFrame([
        {"date": "2024-08-10", "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 1, "result": "D"},
    ])
    result = build_features(matches)
    assert result.loc[0, "target"] == "D"
    assert "home_goals" not in result.columns
    assert "away_goals" not in result.columns


def test_same_day_results_do_not_update_features_until_next_date():
    matches = pd.DataFrame([
        {"date": "2024-08-10", "home_team": "A", "away_team": "B",
         "home_goals": 4, "away_goals": 0, "result": "H"},
        # Synthetic duplicate-date appearance for A deliberately stress-tests leakage handling.
        {"date": "2024-08-10", "home_team": "A", "away_team": "C",
         "home_goals": 0, "away_goals": 2, "result": "A"},
        {"date": "2024-08-17", "home_team": "A", "away_team": "D",
         "home_goals": 1, "away_goals": 0, "result": "H"},
    ])
    result = build_features(matches)
    # Neither fixture on 10 August may use the other's result.
    same_day = result[result["date"] == pd.Timestamp("2024-08-10")]
    assert same_day["home_goals_for_avg"].tolist() == [1.2, 1.2]
    # On the later date, both earlier results are now historical information.
    later = result[result["date"] == pd.Timestamp("2024-08-17")].iloc[0]
    assert later["home_goals_for_avg"] == 2.0
    assert later["home_points_avg"] == 1.5
