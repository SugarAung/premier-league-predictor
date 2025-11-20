import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path("data/processed/match_features_with_elo.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "form_elo_model.pkl"


def train_model_with_elo():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run build_elo_features.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Drop matches that don't have a result/target yet (future fixtures)
    df = df.dropna(subset=["target"])

    # Sort chronologically (important: no future info in train set)
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    feature_cols = [
        "home_avg_goals_for_last5",
        "home_avg_goals_against_last5",
        "home_avg_points_last5",
        "away_avg_goals_for_last5",
        "away_avg_goals_against_last5",
        "away_avg_points_last5",
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
    ]

    # Fill any remaining NaNs in features (early-season matches)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols]
    y = df["target"]  # 'H', 'D', 'A'

    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Total matches: {n}")
    print(f"Training on: {len(X_train)}")
    print(f"Testing on: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight=None,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy on test set:", round(acc, 3))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=["H", "D", "A"]))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": clf,
                "feature_cols": feature_cols,
            },
            f,
        )

    print(f"\nModel with Elo saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model_with_elo()
