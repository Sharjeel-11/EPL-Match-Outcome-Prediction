"""Download, standardise and validate EPL match results."""
from __future__ import annotations

from pathlib import Path
import time
import pandas as pd
import requests

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Completed seasons only, from 2010/11 through 2024/25.
SEASON_CODES = [
    "1011", "1112", "1213", "1314", "1415",
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425",
]
BASE_URL = "https://datahub.io/football/english-premier-league/_r/-/season-{season}.csv"

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def download_seasons(
    season_codes: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
    pause_seconds: float = 0.15,
) -> list[Path]:
    """Download season CSVs, reusing local files when already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    codes = season_codes or SEASON_CODES
    paths = []
    session = requests.Session()
    session.headers.update({"User-Agent": "EPL-match-outcome-portfolio-project/1.0"})
    for code in codes:
        path = raw_dir / f"season-{code}.csv"
        if not path.exists() or path.stat().st_size == 0:
            url = BASE_URL.format(season=code)
            response = session.get(url, timeout=30)
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError(f"Downloaded an empty file for season {code}")
            path.write_text(response.text, encoding="utf-8")
            time.sleep(pause_seconds)
        paths.append(path)
    return paths


def load_matches(raw_dir: Path = RAW_DIR, season_codes: list[str] | None = None) -> pd.DataFrame:
    """Load, standardise and validate the downloaded season files."""
    codes = season_codes or SEASON_CODES
    frames = []
    for code in codes:
        path = raw_dir / f"season-{code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run download_seasons() first.")
        frame = pd.read_csv(path)
        missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frame = frame[REQUIRED_COLUMNS].copy()
        frame["season_code"] = code
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
    })
    # Parse ISO dates explicitly first (the current source uses YYYY-MM-DD).
    # Only fall back to day-first parsing for rows that are not ISO date strings.
    # Do not use dayfirst=True on ISO strings: it can silently swap components
    # in ambiguous dates and shift matches into the wrong season.
    raw_dates = matches["date"].astype("string").str.strip()
    iso_mask = raw_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    parsed_dates = pd.Series(pd.NaT, index=matches.index, dtype="datetime64[ns]")
    parsed_dates.loc[iso_mask] = pd.to_datetime(
        raw_dates.loc[iso_mask], format="%Y-%m-%d", errors="coerce"
    )
    fallback_mask = (~iso_mask) & raw_dates.notna()
    parsed_dates.loc[fallback_mask] = pd.to_datetime(
        raw_dates.loc[fallback_mask], dayfirst=True, errors="coerce"
    )
    matches["date"] = parsed_dates
    for col in ["home_goals", "away_goals"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches["result"] = matches["result"].astype("string").str.strip().str.upper()
    matches["home_team"] = matches["home_team"].astype("string").str.strip()
    matches["away_team"] = matches["away_team"].astype("string").str.strip()

    matches = matches.dropna(subset=[
        "date", "home_team", "away_team", "home_goals", "away_goals", "result"
    ])
    matches = matches[
        matches["result"].isin(["H", "D", "A"])
        & (matches["home_team"] != "")
        & (matches["away_team"] != "")
    ].copy()
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)

    # The result label should agree with the full-time score.
    expected = matches.apply(
        lambda r: "H" if r.home_goals > r.away_goals
        else ("A" if r.home_goals < r.away_goals else "D"),
        axis=1,
    )
    matches = matches.loc[expected.eq(matches["result"])].copy()

    matches = matches.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    )
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    matches["target"] = matches["result"].map({"H": "Home win", "D": "Draw", "A": "Away win"})
    matches["season"] = matches["season_code"].map(_season_label)
    return matches


def _season_label(code: str) -> str:
    """Convert a four-digit season code such as 2425 to 2024/25."""
    start = 2000 + int(code[:2]) if int(code[:2]) < 80 else 1900 + int(code[:2])
    end = (start + 1) % 100
    return f"{start}/{end:02d}"


def prepare_data() -> pd.DataFrame:
    """Download configured seasons and return validated matches."""
    download_seasons()
    matches = load_matches()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)
    return matches
