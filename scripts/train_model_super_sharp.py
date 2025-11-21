import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/match_features_super.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_super_sharp.pkl"   # <-- NEW NAME


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run build_super_features.py first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def train_super_sharp():
    df = load_data()

    # Only use matches with full market probabilities (high-quality odds)
    df = df.dropna(
        subset=["target", "prob_home_market", "prob_draw_market", "prob_away_market"]
    )

    # Sort chronologically
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    feature_cols = [
        # Form
        "home_avg_goals_for_last5",
        "home_avg_goals_against_last5",
        "home_avg_points_last5",
        "away_avg_goals_for_last5",
        "away_avg_goals_against_last5",
        "away_avg_points_last5",
        # Elo
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
        "elo_ratio_home",
        "elo_ratio_away",
        # Odds + probs
        "home_odds",
        "draw_odds",
        "away_odds",
        "prob_home_market",
        "prob_draw_market",
        "prob_away_market",
        "odds_imbalance_home_away",
    ]

    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols]
    y = df["target"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]
    y_test_labels = y.iloc[split_idx:]

    print(f"Total sharp matches: {n}")
    print(f"Training on: {len(X_train)}")
    print(f"Testing on: {len(X_test)}")

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
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
    print("\nAccuracy on sharp test set:", round(acc, 3))
    print("\nClassification report:")
    print(classification_report(y_test_labels, y_pred))
    print("Confusion matrix (rows=true, cols=pred):")
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

    print(f"\nSharp XGBoost model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_super_sharp()
