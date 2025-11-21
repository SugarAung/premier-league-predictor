import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "models/xgb_super_hybrid.pkl"
FEATS_PATH = "data/processed/match_features_super.csv"


def build_design_matrices(df, full_model_bundle, sharp_model_bundle):
    """Same logic as in backtest_hybrid.py, but copied here."""
    full_feats = full_model_bundle["feature_cols"]
    sharp_feats = sharp_model_bundle["feature_cols"]

    missing_full = [c for c in full_feats if c not in df.columns]
    missing_sharp = [c for c in sharp_feats if c not in df.columns]

    if missing_full:
        raise ValueError(
            "Features CSV is missing columns required by FULL model:\n"
            f"{missing_full}"
        )
    if missing_sharp:
        raise ValueError(
            "Features CSV is missing columns required by SHARP model:\n"
            f"{missing_sharp}"
        )

    X_full = df[full_feats].copy().fillna(0.0).values
    X_sharp = df[sharp_feats].copy().fillna(0.0).values
    return X_full, X_sharp


def hybrid_predict_proba(full_model_bundle, sharp_model_bundle, X_full, X_sharp, alpha=0.75):
    """Same blending logic as in backtest_hybrid.py."""
    full_model = full_model_bundle["model"]
    sharp_model = sharp_model_bundle["model"]

    proba_full = full_model.predict_proba(X_full)
    proba_sharp = sharp_model.predict_proba(X_sharp)

    if proba_full.shape != proba_sharp.shape:
        raise ValueError(
            f"Shape mismatch between full ({proba_full.shape}) and sharp "
            f"({proba_sharp.shape}) probabilities."
        )

    return alpha * proba_sharp + (1.0 - alpha) * proba_full


def main():
    # 1. Load hybrid model
    with open(MODEL_PATH, "rb") as f:
        hybrid = pickle.load(f)

    full_bundle = hybrid["full_model"]
    sharp_bundle = hybrid["sharp_model"]
    label_encoder = hybrid["label_encoder"]
    print("Label encoder classes_:", list(label_encoder.classes_))

    # 2. Load a few rows from current season
    df = pd.read_csv(FEATS_PATH, parse_dates=["date"])
    df = df[df["season"] == "2025-2026"].head(5).copy()
    print("Sample rows from 2025-2026:", len(df))

    # 3. Design matrices
    X_full, X_sharp = build_design_matrices(df, full_bundle, sharp_bundle)

    # 4. Raw probabilities
    full_model = full_bundle["model"]
    sharp_model = sharp_bundle["model"]

    proba_full = full_model.predict_proba(X_full)
    proba_sharp = sharp_model.predict_proba(X_sharp)
    proba_hybrid = hybrid_predict_proba(full_bundle, sharp_bundle, X_full, X_sharp)

    print("\nproba_full:\n", proba_full)
    print("\nproba_sharp:\n", proba_sharp)
    print("\nproba_hybrid:\n", proba_hybrid)

    print(
        "\nAny NaNs?",
        "full:", np.isnan(proba_full).any(),
        "sharp:", np.isnan(proba_sharp).any(),
        "hybrid:", np.isnan(proba_hybrid).any(),
    )


if __name__ == "__main__":
    main()
