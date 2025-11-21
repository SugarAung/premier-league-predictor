#!/usr/bin/env python
"""
Backtest the hybrid XGBoost model on historical matches.

Typical usage (from project root):
    python scripts/backtest_hybrid.py

This will:
- load the hybrid model from models/xgb_super_hybrid.pkl
- load historical features from data/processed/match_features_super.csv
- filter to the current season (2025-2026) by default
- write backtest results to data/processed/backtest_hybrid.csv

The Streamlit app reads that CSV to show the Backtest page.
"""

import os
import argparse
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd


# --------------------------------------------------
# Helpers to load model + data
# --------------------------------------------------
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


def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Features CSV is empty – nothing to backtest on.")
    if "result" not in df.columns:
        raise ValueError("Features CSV must contain a 'result' column (H/D/A).")

    # Only keep matches with a real result
    before = len(df)
    df = df[df["result"].isin(["H", "D", "A"])].copy()
    print(f"[backtest] Filtered to matches with known result: {before} -> {len(df)} rows")

    return df


def build_design_matrices(
    df: pd.DataFrame,
    full_model_bundle: Dict[str, Any],
    sharp_model_bundle: Dict[str, Any],
):
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


def hybrid_predict_proba(
    full_model_bundle: Dict[str, Any],
    sharp_model_bundle: Dict[str, Any],
    X_full,
    X_sharp,
    alpha: float = 0.75,
):
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

    df = pd.DataFrame(
        {
            "prob_H": prob_H,
            "prob_D": prob_D,
            "prob_A": prob_A,
            "predicted_label": pred_labels,
        }
    )
    return df


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Backtest hybrid model on historical matches."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/xgb_super_hybrid.pkl",
        help="Path to hybrid model pickle.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/match_features_super.csv",
        help="Path to historical features CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/backtest_hybrid.csv",
        help="Where to save backtest results CSV.",
    )
    parser.add_argument(
        "--min_season",
        type=str,
        default="2025-2026",
        help="Minimum season string to include (e.g. '2018-2019'). "
             "Default = '2025-2026' (current season only).",
    )

    args = parser.parse_args()

    print(f"[backtest] Loading hybrid model from: {args.model}")
    hybrid = load_hybrid_model(args.model)
    full_bundle = hybrid["full_model"]
    sharp_bundle = hybrid["sharp_model"]
    label_encoder = hybrid["label_encoder"]

    print(f"[backtest] Loading historical features from: {args.features}")
    df = load_features(args.features)

    # Filter by season if requested
    if args.min_season is not None:
        before_rows = len(df)
        df = df[df["season"] >= args.min_season].copy()
        print(
            f"[backtest] Filtered seasons >= {args.min_season}: "
            f"{before_rows} -> {len(df)} rows"
        )

    if df.empty:
        raise ValueError("[backtest] No rows left after season filtering – nothing to backtest.")

    # 🔧 IMPORTANT: reset index so it lines up with prob_df
    df = df.reset_index(drop=True)

    print("[backtest] Building design matrices...")
    X_full, X_sharp = build_design_matrices(df, full_bundle, sharp_bundle)

    print("[backtest] Predicting hybrid probabilities for historical matches...")
    proba = hybrid_predict_proba(full_bundle, sharp_bundle, X_full, X_sharp, alpha=0.75)

    print("[backtest] Converting probabilities to labels...")
    prob_df = proba_to_labels(proba, label_encoder)
    prob_df = prob_df.reset_index(drop=True)

    # Attach to original df (row-wise, ignoring index labels)
    out = df.copy()
    for col in prob_df.columns:
        out[col] = prob_df[col].values

    # Evaluation columns
    out["actual_result"] = out["result"]
    out["correct"] = out["predicted_label"] == out["actual_result"]

    overall_acc = out["correct"].mean() * 100.0
    print(
        f"\n[backtest] Overall accuracy on backtest set: {overall_acc:.2f}% "
        f"({out['correct'].sum()} / {len(out)} matches)"
    )

    # Accuracy by gameweek (for this season)
    if "season" in out.columns and "gameweek" in out.columns:
        acc_by_gw = out.groupby(["season", "gameweek"])["correct"].mean() * 100.0
        print("\n[backtest] Accuracy by season/gameweek:")
        for (season, gw), acc in acc_by_gw.sort_index().items():
            print(f"  {season} GW{int(gw)}: {acc:.2f}%")

    # Save file
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n[backtest] Saved backtest results to: {args.output}")
    print("[backtest] Done.")


if __name__ == "__main__":
    main()
