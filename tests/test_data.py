import pandas as pd

from src.data import load_matches


def test_iso_dates_are_parsed_without_swapping_month_and_day(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        [
            {"Date": "2010-08-14", "HomeTeam": "Aston Villa", "AwayTeam": "West Ham",
             "FTHG": 3, "FTAG": 0, "FTR": "H"},
            {"Date": "2025-05-25", "HomeTeam": "Liverpool", "AwayTeam": "Crystal Palace",
             "FTHG": 1, "FTAG": 1, "FTR": "D"},
        ]
    ).to_csv(raw_dir / "season-1011.csv", index=False)

    matches = load_matches(raw_dir=raw_dir, season_codes=["1011"])

    assert matches["date"].min() == pd.Timestamp("2010-08-14")
    assert matches["date"].max() == pd.Timestamp("2025-05-25")


def test_non_iso_day_first_dates_are_still_supported(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        [
            {"Date": "14/08/2010", "HomeTeam": "Aston Villa", "AwayTeam": "West Ham",
             "FTHG": 3, "FTAG": 0, "FTR": "H"},
        ]
    ).to_csv(raw_dir / "season-1011.csv", index=False)

    matches = load_matches(raw_dir=raw_dir, season_codes=["1011"])

    assert matches.loc[0, "date"] == pd.Timestamp("2010-08-14")
