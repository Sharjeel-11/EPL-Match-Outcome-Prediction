"""Build leakage-aware pre-match features from historical team results."""
from __future__ import annotations

import pandas as pd


def build_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build team-form predictors using only matches on earlier dates.

    All fixtures on the same date are featurised before any results from that
    date are added to team history. This prevents same-day result leakage even
    if a future dataset contains multiple fixtures for one team on one date.
    Cold-start values are explicit defaults rather than rolling-window nulls.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals", "result"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if window < 1:
        raise ValueError("window must be at least 1")

    ordered = matches.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    if ordered["date"].isna().any():
        raise ValueError("Matches contain missing or invalid dates")
    ordered = ordered.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    history: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, object]] = []

    def summarise(team: str) -> dict[str, float]:
        prior = history.get(team, [])[-window:]
        if not prior:
            return {
                "points_avg": 1.0,
                "goals_for_avg": 1.2,
                "goals_against_avg": 1.2,
                "goal_difference_avg": 0.0,
                "win_rate": 1 / 3,
                "draw_rate": 1 / 3,
            }
        frame = pd.DataFrame(prior)
        return {
            "points_avg": float(frame["points"].mean()),
            "goals_for_avg": float(frame["goals_for"].mean()),
            "goals_against_avg": float(frame["goals_against"].mean()),
            "goal_difference_avg": float(frame["goal_difference"].mean()),
            "win_rate": float((frame["points"] == 3).mean()),
            "draw_rate": float((frame["points"] == 1).mean()),
        }

    # Create predictors for a whole date first, then update history afterwards.
    for _, day_matches in ordered.groupby("date", sort=True):
        day_updates: list[tuple[str, str, float, float]] = []
        for _, match in day_matches.iterrows():
            home, away = str(match["home_team"]), str(match["away_team"])
            home_stats = summarise(home)
            away_stats = summarise(away)
            row: dict[str, object] = {
                "date": match["date"],
                "season": match.get("season", ""),
                "home_team": home,
                "away_team": away,
                "target": str(match["result"]),
            }
            for key, value in home_stats.items():
                row[f"home_{key}"] = value
            for key, value in away_stats.items():
                row[f"away_{key}"] = value
            for key in home_stats:
                row[f"form_diff_{key}"] = home_stats[key] - away_stats[key]
            rows.append(row)

            hg, ag = float(match["home_goals"]), float(match["away_goals"])
            day_updates.append((home, away, hg, ag))
        for home, away, hg, ag in day_updates:
            if hg > ag:
                hp, ap = 3.0, 0.0
            elif hg < ag:
                hp, ap = 0.0, 3.0
            else:
                hp, ap = 1.0, 1.0
            history.setdefault(home, []).append({
                "points": hp, "goals_for": hg, "goals_against": ag,
                "goal_difference": hg - ag,
            })
            history.setdefault(away, []).append({
                "points": ap, "goals_for": ag, "goals_against": hg,
                "goal_difference": ag - hg,
            })

    features = pd.DataFrame(rows)
    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].replace([float("inf"), -float("inf")], pd.NA)
    return features
