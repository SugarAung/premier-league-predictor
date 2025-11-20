#!/usr/bin/env python
"""
Use the trained hybrid XGBoost models to predict outcomes
(H / D / A) for upcoming fixtures.

Usage (defaults should work):
    python predict_hybrid.py

Or explicitly:
    python predict_hybrid.py \
        --model models/xgb_super_hybrid.pkl \
        --fixtures data/processed/pl_fixtures_features.csv \
        --output pl_predictions.csv
"""

import os
import argparse
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd


def load_hybrid_model(path: str) -> Dict[str, Any]:
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fixtures features file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Fixtures features CSV is empty – nothing to predict on.")
    return df


def build_design_matrices(
    fixtures_df: pd.DataFrame,
    full_model_bundle: Dict[str, Any],
    sharp_model_bundle: Dict[str, Any],
) -> (np.ndarray, np.ndarray):
    """
    Build X_full and X_sharp using the stored feature_col lists
    inside each model bundle. Fill NaNs with 0.0 like in training.
    """
    full_feats = full_model_bundle["feature_cols"]
    sharp_feats = sharp_model_bundle["feature_cols"]

    missing_full = [c for c in full_feats if c not in fixtures_df.columns]
    missing_sharp = [c for c in sharp_feats if c not in fixtures_df.columns]

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

    X_full = fixtures_df[full_feats].copy().fillna(0.0)
    X_sharp = fixtures_df[sharp_feats].copy().fillna(0.0)

    return X_full.values, X_sharp.values


def hybrid_predict_proba(
    full_model_bundle: Dict[str, Any],
    sharp_model_bundle: Dict[str, Any],
    X_full: np.ndarray,
    X_sharp: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Weighted ensemble:
        final_prob = alpha * sharp + (1 - alpha) * full
    alpha = 0.75 (same as in train_model_super_hybrid.py)
    """
    full_model = full_model_bundle["model"]
    sharp_model = sharp_model_bundle["model"]

    if not hasattr(full_model, "predict_proba"):
        raise ValueError("Full model does not support predict_proba().")
    if not hasattr(sharp_model, "predict_proba"):
        raise ValueError("Sharp model does not support predict_proba().")

    proba_full = full_model.predict_proba(X_full)
    proba_sharp = sharp_model.predict_proba(X_sharp)

    if proba_full.shape != proba_sharp.shape:
        raise ValueError(
            f"Shape mismatch between full ({proba_full.shape}) and sharp "
            f"({proba_sharp.shape}) probabilities."
        )

    return alpha * proba_sharp + (1.0 - alpha) * proba_full


def proba_to_labels(
    proba: np.ndarray,
    label_encoder,
) -> pd.DataFrame:
    """
    Convert probability matrix (n_samples x n_classes) into
    prob_H / prob_D / prob_A + predicted_label using label_encoder.classes_.
    """
    classes = list(label_encoder.classes_)  # e.g. ['A', 'D', 'H']
    n_classes = len(classes)

    if proba.shape[1] != n_classes:
        raise ValueError(
            f"Probability array has {proba.shape[1]} columns, "
            f"but label_encoder has {n_classes} classes: {classes}"
        )

    label_to_index = {label: idx for idx, label in enumerate(classes)}

    # Get column indices for each result
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

    # Predicted encoded class and labels
    pred_enc = proba.argmax(axis=1)
    pred_labels = [classes[i] for i in pred_enc]

    df = pd.DataFrame(
        {
            "prob_H": prob_H,
            "prob_D": prob_D,
            "prob_A": prob_A,
            "predicted_label": pred_labels,
        }
    )
    return df


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
        default="pl_predictions.csv",
        help="Output CSV file for predictions.",
    )

    args = parser.parse_args()

    print(f"Loading hybrid model from: {args.model}")
    hybrid = load_hybrid_model(args.model)
    full_bundle = hybrid["full_model"]
    sharp_bundle = hybrid["sharp_model"]
    label_encoder = hybrid["label_encoder"]

    print(f"Loading fixtures features from: {args.fixtures}")
    fixtures_df = load_fixtures_features(args.fixtures)

    print("Building design matrices for full + sharp models...")
    X_full, X_sharp = build_design_matrices(fixtures_df, full_bundle, sharp_bundle)

    print("Predicting hybrid probabilities...")
    proba = hybrid_predict_proba(full_bundle, sharp_bundle, X_full, X_sharp, alpha=0.75)

    print("Converting probabilities to labels...")
    prob_df = proba_to_labels(proba, label_encoder)

    # Merge with original fixtures info
    output_df = fixtures_df.copy()
    for col in prob_df.columns:
        output_df[col] = prob_df[col]

    # Human-readable label
    outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    output_df["predicted_outcome"] = output_df["predicted_label"].map(outcome_map)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")
    print(f"Rows: {len(output_df)}")
    print("Columns:", list(output_df.columns))


if __name__ == "__main__":
    main()
