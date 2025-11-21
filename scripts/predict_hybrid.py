#!/usr/bin/env python
"""
Use the trained hybrid XGBoost models to predict outcomes
(H / D / A) for upcoming fixtures.

Typical usage (from project root):
    python scripts/predict_hybrid.py

This will:
- load the hybrid model from models/xgb_super_hybrid.pkl
- load fixture features from data/processed/pl_fixtures_features.csv
- write predictions to data/live/pl_predictions.csv
"""

import os
import argparse
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd


def load_hybrid_model(path: str) -> Dict[str, Any]:
    """Load the hybrid model dict with full_model, sharp_model, label_encoder."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hybrid model file not found: {path}")
    with open(path, "rb") as f:
        hybrid = pickle.load(f)

    required_keys = {"full_model", "sharp_model", "label_encoder"}
    if not required_keys.issubset(hybrid.keys()):
        raise ValueError(
            f"Hybrid model dict must contain keys {required_keys}, "
            f"found {set(hybrid.keys())}"
        )
    return hybrid


def load_fixtures_features(path: str) -> pd.DataFrame:
    """Load engineered features for upcoming fixtures."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fixtures features file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Fixtures features CSV is empty – nothing to predict on.")
    return df


def build_design_matrices(
    df: pd.DataFrame,
    full_model_bundle: Dict[str, Any],
    sharp_model_bundle: Dict[str, Any],
):
    """Build X matrices for full and sharp models using their feature_cols."""
    full_feats = full_model_bundle["feature_cols"]
    sharp_feats = sharp_model_bundle["feature_cols"]

    missing_full = [c for c in full_feats if c not in df.columns]
    missing_sharp = [c for c in sharp_feats if c not in df.columns]

    if missing_full:
        raise ValueError(
            "Fixtures features CSV is missing columns required by FULL model:\n"
            f"{missing_full}"
        )
    if missing_sharp:
        raise ValueError(
            "Fixtures features CSV is missing columns required by SHARP model:\n"
            f"{missing_sharp}"
        )

    X_full = df[full_feats].copy().fillna(0.0).values
    X_sharp = df[sharp_feats].copy().fillna(0.0).values

    return X_full, X_sharp


def hybrid_predict_proba(
    full_model_bundle: Dict[str, Any],
    sharp_model_bundle: Dict[str, Any],
    X_full,
    X_sharp,
    alpha: float = 0.75,
):
    """Blend probabilities from sharp + full models: alpha*sharp + (1-alpha)*full."""
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


def proba_to_labels(proba: np.ndarray, label_encoder) -> pd.DataFrame:
    """Convert probability matrix → prob_H/prob_D/prob_A + predicted_label."""
    classes = list(label_encoder.classes_)  # e.g. ['A', 'D', 'H']
    n_classes = len(classes)

    if proba.shape[1] != n_classes:
        raise ValueError(
            f"Probability array has {proba.shape[1]} columns, "
            f"but label_encoder has {n_classes} classes: {classes}"
        )

    label_to_index = {label: idx for idx, label in enumerate(classes)}

    try:
        idx_H = label_to_index["H"]
        idx_D = label_to_index["D"]
        idx_A = label_to_index["A"]
    except KeyError:
        raise ValueError(
            f"Label encoder classes must include 'H', 'D', 'A', got {classes}"
        )

    prob_H = proba[:, idx_H]
    prob_D = proba[:, idx_D]
    prob_A = proba[:, idx_A]

    pred_enc = proba.argmax(axis=1)
    pred_labels = [classes[i] for i in pred_enc]

    return pd.DataFrame(
        {
            "prob_H": prob_H,
            "prob_D": prob_D,
            "prob_A": prob_A,
            "predicted_label": pred_labels,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict PL match outcomes using the hybrid XGBoost ensemble."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/xgb_super_hybrid.pkl",
        help="Path to the hybrid model pickle file.",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default="data/processed/pl_fixtures_features.csv",
        help="Path to CSV with upcoming fixtures (engineered features).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/live/pl_predictions.csv",
        help="Output CSV file for predictions.",
    )

    args = parser.parse_args()

    print(f"[predict] Loading hybrid model from: {args.model}")
    hybrid = load_hybrid_model(args.model)
    full_bundle = hybrid["full_model"]
    sharp_bundle = hybrid["sharp_model"]
    label_encoder = hybrid["label_encoder"]

    print(f"[predict] Loading fixture features from: {args.fixtures}")
    df = load_fixtures_features(args.fixtures)

    print("[predict] Building design matrices for full + sharp models...")
    X_full, X_sharp = build_design_matrices(df, full_bundle, sharp_bundle)

    print("[predict] Predicting hybrid probabilities for upcoming fixtures...")
    proba = hybrid_predict_proba(full_bundle, sharp_bundle, X_full, X_sharp, alpha=0.75)

    print("[predict] Converting probabilities to labels...")
    prob_df = proba_to_labels(proba, label_encoder)

    # Attach metadata (season, gameweek, date, home/away) + probs + label
    meta_cols = ["season", "gameweek", "date", "home_team", "away_team"]
    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise ValueError(
            "Fixtures features CSV is missing required metadata columns:\n"
            f"{missing_meta}"
        )

    out = df[meta_cols].copy()
    for col in prob_df.columns:
        out[col] = prob_df[col]

    # Ensure pred_result column for app
    out["pred_result"] = out["predicted_label"]

    # Make sure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out.to_csv(args.output, index=False)

    print(f"[predict] Saved predictions for {len(out)} fixtures to: {args.output}")
    print("[predict] Columns:", list(out.columns))


if __name__ == "__main__":
    main()
