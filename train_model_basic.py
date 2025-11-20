import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
DATA_PATH = Path("data/processed/match_features_basic.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "basic_form_model.pkl"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run build_features_basic.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def train_basic_model():
    df = load_data()

    # Drop matches that don't have a result/target yet (future fixtures)
    df = df.dropna(subset=["target"])

    # Sort chronologically (important: no future info in train set)
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    # Feature columns we will use for this first model
    feature_cols = [
        "home_avg_goals_for_last5",
        "home_avg_goals_against_last5",
        "home_avg_points_last5",
        "away_avg_goals_for_last5",
        "away_avg_goals_against_last5",
        "away_avg_points_last5",
    ]

    # Safety: fill any remaining NaNs in features (early matches with no history)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols]
    y = df["target"]  # 'H', 'D', 'A'

    # Simple time-based split: first 80% train, last 20% test
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Total matches: {n}")
    print(f"Training on: {len(X_train)}")
    print(f"Testing on: {len(X_test)}")

    # ---------------------------------------------------------
    # Train a basic Logistic Regression model
    # ---------------------------------------------------------
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
    )
    clf.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Evaluate on the test set
    # ---------------------------------------------------------
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy on test set:", round(acc, 3))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, y_pred, labels=["H", "D", "A"]))

    # ---------------------------------------------------------
    # Save the trained model to disk
    # ---------------------------------------------------------
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": clf,
                "feature_cols": feature_cols,
            },
            f,
        )

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_basic_model()
