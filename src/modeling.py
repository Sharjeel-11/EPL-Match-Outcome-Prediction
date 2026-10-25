"""Chronological evaluation and model comparison."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "home_points_avg", "home_goals_for_avg", "home_goals_against_avg",
    "home_goal_difference_avg", "home_win_rate", "home_draw_rate",
    "away_points_avg", "away_goals_for_avg", "away_goals_against_avg",
    "away_goal_difference_avg", "away_win_rate", "away_draw_rate",
    "form_diff_points_avg", "form_diff_goals_for_avg",
    "form_diff_goals_against_avg", "form_diff_goal_difference_avg",
    "form_diff_win_rate", "form_diff_draw_rate",
]
CLASS_ORDER = ["H", "D", "A"]


def add_season_year(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["season_start"] = frame["date"].dt.year.where(
        frame["date"].dt.month >= 7, frame["date"].dt.year - 1
    )
    return frame


def temporal_split(features: pd.DataFrame):
    """Train through 2020/21, validate on 2021/22-2022/23, test on 2023/24-2024/25."""
    frame = add_season_year(features)
    train = frame[frame["season_start"] <= 2020].copy()
    valid = frame[frame["season_start"].isin([2021, 2022])].copy()
    test = frame[frame["season_start"].isin([2023, 2024])].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(
            "One temporal split is empty. Check that data includes seasons through 2024/25."
        )
    return train, valid, test


def make_models():
    return {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight=None, random_state=42)),
        ]),
        "Random forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=8, max_features="sqrt",
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )),
        ]),
    }


def score_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compare_models(features: pd.DataFrame, output_dir: Path = Path("reports")):
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = temporal_split(features)
    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_valid, y_valid = valid[FEATURE_COLUMNS], valid["target"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target"]

    validation_rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        validation_rows.append({"model": name, **score_predictions(y_valid, pred)})
    validation = pd.DataFrame(validation_rows).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    )
    validation.to_csv(output_dir / "validation_metrics.csv", index=False)
    best_name = str(validation.iloc[0]["model"])

    # After selecting the model on validation data, refit using train + validation.
    # The final test seasons are not used to choose a model or tune parameters.
    development = pd.concat([train, valid], ignore_index=True)
    selected = make_models()[best_name]
    selected.fit(development[FEATURE_COLUMNS], development["target"])
    test_pred = selected.predict(X_test)
    test_scores = score_predictions(y_test, test_pred)

    # Evaluate a fitted majority-class baseline on the exact same test matches.
    baseline = make_models()["Majority-class baseline"]
    baseline.fit(development[FEATURE_COLUMNS], development["target"])
    baseline_pred = baseline.predict(X_test)
    baseline_scores = score_predictions(y_test, baseline_pred)

    if best_name == "Majority-class baseline":
        test_comparison = [
            {"model": "Majority-class baseline (selected)", **test_scores}
        ]
    else:
        test_comparison = [
            {"model": "Majority-class baseline", **baseline_scores},
            {"model": f"{best_name} (selected)", **test_scores},
        ]
    pd.DataFrame(test_comparison).to_csv(output_dir / "test_metrics.csv", index=False)

    test_results = {
        "selected_model_by_validation_macro_f1": best_name,
        "validation_seasons": ["2021/22", "2022/23"],
        "test_seasons": ["2023/24", "2024/25"],
        "test_metrics": test_scores,
        "test_baseline_metrics": baseline_scores,
        "test_model_comparison": test_comparison,
        "test_rows": int(len(test)),
        "classification_report": classification_report(
            y_test, test_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
        ),
    }
    (output_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8"
    )

    cm = confusion_matrix(y_test, test_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Home win", "Draw", "Away win"],
        yticklabels=["Home win", "Draw", "Away win"],
    )
    plt.xlabel("Predicted outcome")
    plt.ylabel("Actual outcome")
    plt.title(f"Test-set confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close()

    return validation, test_results, selected, (train, valid, test), test_pred
