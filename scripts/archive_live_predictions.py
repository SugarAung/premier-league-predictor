import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Config (paths are relative to project root)
# --------------------------------------------------
PRED_PATH = Path("data/live/pl_predictions.csv")              # live model predictions
RESULTS_PATH = Path("data/live/pl_results_from_api.csv")      # actual results from API
ARCHIVE_PATH = Path("data/live/live_predictions_history.csv") # permanent archive (one row per match)


def main():
    # -------------------------------
    # 1. Load predictions & results
    # -------------------------------
    if not PRED_PATH.exists():
        print(f"[archive] No {PRED_PATH} found. Nothing to archive.")
        return

    if not RESULTS_PATH.exists():
        print(f"[archive] No {RESULTS_PATH} found. Cannot match predictions to results.")
        return

    print(f"[archive] Loading predictions from: {PRED_PATH}")
    pred_df = pd.read_csv(PRED_PATH)

    print(f"[archive] Loading results from: {RESULTS_PATH}")
    results_df = pd.read_csv(RESULTS_PATH)

    # We only archive matches that:
    # - have an actual_result
    # - have a prediction (predicted_label or pred_result)
    if "predicted_label" in pred_df.columns:
        pred_df["pred_result"] = pred_df["predicted_label"]
    elif "pred_result" not in pred_df.columns:
        pred_df["pred_result"] = ""

    # Merge on match key
    key_cols = ["season", "gameweek", "date", "home_team", "away_team"]
    missing_pred = [c for c in key_cols if c not in pred_df.columns]
    missing_res = [c for c in key_cols if c not in results_df.columns]
    if missing_pred:
        raise ValueError(f"[archive] Predictions CSV missing key columns: {missing_pred}")
    if missing_res:
        raise ValueError(f"[archive] Results CSV missing key columns: {missing_res}")

    merged = results_df.merge(
        pred_df,
        on=key_cols,
        how="left",
        suffixes=("_res", "_pred"),
    )

    # Normalise actual_result column name
    if "actual_result_res" in merged.columns:
        merged["actual_result"] = merged["actual_result_res"]
    elif "actual_result" not in merged.columns:
        raise ValueError("[archive] Results CSV must contain 'actual_result' column.")

    # Only keep matches with actual result + prediction
    merged["pred_result"] = merged["pred_result"].fillna("").astype(str).str.strip()
    has_pred = merged["pred_result"] != ""
    has_actual = merged["actual_result"].notna()

    archive_candidates = merged[has_pred & has_actual].copy()
    if archive_candidates.empty:
        print("[archive] No finished matches with predictions to archive.")
        return

    # Compute correctness if not there
    if "correct" not in archive_candidates.columns:
        archive_candidates["correct"] = archive_candidates["pred_result"] == archive_candidates["actual_result"]

    # Pick columns to store in archive (match key + prediction/probs/flags)
    archive_cols = key_cols + [
        "pred_result",
        "actual_result",
        "correct",
    ]
    for col in ["prob_H", "prob_D", "prob_A"]:
        if col in archive_candidates.columns:
            archive_cols.append(col)

    new_data = archive_candidates[archive_cols].copy()

    # -------------------------------
    # 2. Merge into archive CSV
    # -------------------------------
    if ARCHIVE_PATH.exists():
        archive_df = pd.read_csv(ARCHIVE_PATH)
        print(f"[archive] Loaded existing archive with {len(archive_df)} rows.")
    else:
        archive_df = pd.DataFrame(columns=archive_cols)
        print("[archive] No existing archive, will create a new one.")

    # Avoid duplicates: drop any rows already in archive by match key
    key_cols_for_dedup = key_cols  # one entry per match
    merged_archive = pd.concat([archive_df, new_data], ignore_index=True)

    merged_archive["__dupe_key__"] = merged_archive[key_cols_for_dedup].astype(str).agg("|".join, axis=1)
    merged_archive = merged_archive.drop_duplicates(subset="__dupe_key__", keep="last")
    merged_archive = merged_archive.drop(columns="__dupe_key__")

    # Sort nicely
    sort_cols = [c for c in ["season", "gameweek", "date", "home_team", "away_team"] if c in merged_archive.columns]
    if sort_cols:
        merged_archive = merged_archive.sort_values(sort_cols).reset_index(drop=True)

    # Ensure folder exists
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_archive.to_csv(ARCHIVE_PATH, index=False)

    # Which gameweeks did we just add/refresh?
    new_gws = sorted(new_data["gameweek"].unique()) if "gameweek" in new_data.columns else []

    print(
        f"[archive] Archived/updated {len(new_data)} rows from "
        f"{len(new_gws)} gameweek(s) into {ARCHIVE_PATH}."
    )
    print("[archive] Done.")


if __name__ == "__main__":
    main()
