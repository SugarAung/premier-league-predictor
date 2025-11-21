import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/match_features_super.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_super_model.pkl"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run build_super_features.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def train_super_model():
    df = load_data()

    # 1) Drop ONLY rows with no target (future fixtures)
    df = df.dropna(subset=["target"])

    # 2) Sort chronologically to avoid leakage
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    # Feature set: form + Elo + multi-book odds + probabilities + extras
    feature_cols = [
        # Form features
        "home_avg_goals_for_last5",
        "home_avg_goals_against_last5",
        "home_avg_points_last5",
        "away_avg_goals_for_last5",
        "away_avg_goals_against_last5",
        "away_avg_points_last5",
        # Elo features
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
        "elo_ratio_home",
        "elo_ratio_away",
        # Odds (closing + B365 + combined)
        "home_odds_close",
        "draw_odds_close",
        "away_odds_close",
        "home_odds_b365",
        "draw_odds_b365",
        "away_odds_b365",
        "home_odds",
        "draw_odds",
        "away_odds",
        # Implied probabilities
        "prob_home_close",
        "prob_draw_close",
        "prob_away_close",
        "prob_home_b365",
        "prob_draw_b365",
        "prob_away_b365",
        "prob_home_market",
        "prob_draw_market",
        "prob_away_market",
        "prob_home_raw",
        "prob_draw_raw",
        "prob_away_raw",
        # Extra interaction
        "odds_imbalance_home_away",
    ]

    # 3) Fill missing odds/probs with column means (Option A: keep all matches)
    odds_prob_cols = [
        "home_odds_close",
        "draw_odds_close",
        "away_odds_close",
        "home_odds_b365",
        "draw_odds_b365",
        "away_odds_b365",
        "home_odds",
        "draw_odds",
        "away_odds",
        "prob_home_close",
        "prob_draw_close",
        "prob_away_close",
        "prob_home_b365",
        "prob_draw_b365",
        "prob_away_b365",
        "prob_home_market",
        "prob_draw_market",
        "prob_away_market",
        "prob_home_raw",
        "prob_draw_raw",
        "prob_away_raw",
        "odds_imbalance_home_away",
    ]

    for col in odds_prob_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # 4) Fill any remaining NaNs in other features (e.g. early-season form) with 0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols]
    y = df["target"]  # 'H', 'D', 'A'

    # Encode labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Time-based split
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]
    y_test_labels = y.iloc[split_idx:]

    print(f"Total matches: {n}")
    print(f"Training on: {len(X_train)}")
    print(f"Testing on: {len(X_test)}")

    clf = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)
    y_pred_enc = y_proba.argmax(axis=1)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test_labels, y_pred)
    print("\nAccuracy on test set:", round(acc, 3))

    print("\nClassification report:")
    print(classification_report(y_test_labels, y_pred))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test_labels, y_pred, labels=["H", "D", "A"]))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": clf,
                "label_encoder": le,
                "feature_cols": feature_cols,
            },
            f,
        )

    print(f"\nSuper XGBoost model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_super_model()
