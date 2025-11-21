import pandas as pd
import numpy as np


def main():
    # 1. Read your prediction file and the API results file
    pred_df = pd.read_csv("pl_predictions.csv")
    api_df = pd.read_csv("pl_results_from_api.csv")

    # Keep only the columns we need from the API file
    api_df = api_df[[
        "season",
        "gameweek",
        "home_team",
        "away_team",
        "actual_result",
    ]].copy()

    # 2. Merge predictions (left) with API results (right)
    merged = pred_df.merge(
        api_df,
        on=["season", "gameweek", "home_team", "away_team"],
        how="left",
        suffixes=("", "_api"),
    )

    # 3. Decide where to update actual_result:
    # - only if your current actual_result is empty/NaN
    # - and API has a value in actual_result_api
    actual_is_empty = (
        merged["actual_result"].isna()
        | (merged["actual_result"].astype(str).str.strip() == "")
    )
    api_has_value = merged["actual_result_api"].notna()

    should_update = actual_is_empty & api_has_value

    # 4. Create a new column combining old and new values
    merged["actual_result_new"] = np.where(
        should_update,
        merged["actual_result_api"],
        merged["actual_result"],
    )

    # 5. Replace the old actual_result with the new one
    merged["actual_result"] = merged["actual_result_new"]

    # Drop helper columns
    merged = merged.drop(columns=["actual_result_api", "actual_result_new"])

    # 6. Save to a NEW file so it's safe to inspect
    merged.to_csv("pl_predictions_updated.csv", index=False)

    print("Merge complete!")
    print("Rows in predictions:", len(pred_df))
    print("Rows in merged file:", len(merged))
    print("Updated rows:", int(should_update.sum()))
    print("Saved to pl_predictions_updated.csv")


if __name__ == "__main__":
    main()
