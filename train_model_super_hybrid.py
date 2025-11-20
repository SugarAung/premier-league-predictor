import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# File paths for models
FULL_MODEL_PATH = Path("models/xgb_super_model.pkl")         # full-data model
SHARP_MODEL_PATH = Path("models/xgb_super_sharp.pkl")        # sharp high-odds model

DATA_PATH = Path("data/processed/match_features_super.csv")
OUT_MODEL_PATH = Path("models/xgb_super_hybrid.pkl")


def load_models():
    if not FULL_MODEL_PATH.exists():
        raise FileNotFoundError("Full model not found!")

    if not SHARP_MODEL_PATH.exists():
        raise FileNotFoundError("Sharp model not found!")

    with open(FULL_MODEL_PATH, "rb") as f:
        full = pickle.load(f)

    with open(SHARP_MODEL_PATH, "rb") as f:
        sharp = pickle.load(f)

    return full, sharp


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["target"])  # we need real past results
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    return df


def prepare_features(df, feature_cols):
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df[feature_cols]


def hybrid_predict_proba(full_model, sharp_model, X_full, X_sharp):
    """
    Weighted ensemble:
    final_prob = α * sharp + (1 - α) * full
    α = sharp_weight : we start with 0.75
    """
    α = 0.75

    proba_full = full_model["model"].predict_proba(X_full)
    proba_sharp = sharp_model["model"].predict_proba(X_sharp)

    # weighted ensemble
    return α * proba_sharp + (1 - α) * proba_full


def train_hybrid():
    print("Loading models...")
    full_model, sharp_model = load_models()

    print("Loading data...")
    df = load_data()

    # Use ONLY rows sharp model can predict (same 760 matches)
    df_sharp = df.dropna(
        subset=["prob_home_market", "prob_draw_market", "prob_away_market"]
    ).copy()

    print(f"Hybrid usable matches: {len(df_sharp)}")

    # ----------- Features for each model -----------
    full_cols = full_model["feature_cols"]
    sharp_cols = sharp_model["feature_cols"]

    X_full = prepare_features(df_sharp, full_cols)
    X_sharp = prepare_features(df_sharp, sharp_cols)

    y = df_sharp["target"]

    # Label encoder: use sharp model encoding
    le = sharp_model["label_encoder"]
    y_enc = le.transform(y)

    # Time split
    n = len(df_sharp)
    split_idx = int(n * 0.8)

    X_full_train, X_full_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    X_sharp_train, X_sharp_test = X_sharp.iloc[:split_idx], X_sharp.iloc[split_idx:]

    y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]
    y_test_labels = y.iloc[split_idx:]

    # Hybrid predictions
    print("Computing hybrid predictions...")
    y_proba_test = hybrid_predict_proba(
        full_model, sharp_model, X_full_test, X_sharp_test
    )

    y_pred_enc = y_proba_test.argmax(axis=1)
    y_pred = le.inverse_transform(y_pred_enc)

    # ----------- Evaluation -----------
    print("\nAccuracy on hybrid test set:", round(accuracy_score(y_test_labels, y_pred), 3))

    print("\nClassification report:")
    print(classification_report(y_test_labels, y_pred))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test_labels, y_pred, labels=["H", "D", "A"]))

    # Save hybrid config only (no training, purely ensemble)
    with open(OUT_MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "full_model": full_model,
                "sharp_model": sharp_model,
                "label_encoder": le,
            },
            f,
        )

    print(f"\nHybrid model saved to {OUT_MODEL_PATH}")


if __name__ == "__main__":
    train_hybrid()
