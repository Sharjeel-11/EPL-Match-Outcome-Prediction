"""Run the full EPL match outcome prediction workflow."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import prepare_data
from src.features import build_features
from src.modeling import compare_models, FEATURE_COLUMNS


def main():
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    matches = prepare_data()
    features = build_features(matches, window=5)
    features.to_csv("data/processed/match_features.csv", index=False)

    # Basic dataset diagnostics
    diagnostics = {
        "matches_loaded": int(len(matches)),
        "date_start": str(matches["date"].min().date()),
        "date_end": str(matches["date"].max().date()),
        "unique_teams": int(pd.concat([matches["home_team"], matches["away_team"]]).nunique()),
        "outcome_distribution": matches["result"].value_counts(normalize=True).to_dict(),
        "missing_feature_values": int(features[FEATURE_COLUMNS].isna().sum().sum()),
    }
    (reports / "data_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    matches["result"].value_counts(normalize=True).reindex(["H", "D", "A"]).plot(
        kind="bar", title="Distribution of EPL match outcomes"
    )
    plt.xlabel("Outcome (H=home win, D=draw, A=away win)")
    plt.ylabel("Share of matches")
    plt.tight_layout()
    plt.savefig(reports / "outcome_distribution.png", dpi=180)
    plt.close()

    validation, test_results, selected, splits, test_pred = compare_models(features, reports)
    print("\nData diagnostics")
    print(json.dumps(diagnostics, indent=2))
    print("\nValidation comparison")
    print(validation.to_string(index=False))
    print("\nHeld-out test results")
    print(json.dumps(test_results, indent=2))
    print("\nGenerated files in reports/ and data/processed/")

if __name__ == "__main__":
    main()
