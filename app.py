import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Premier League Predictions", layout="wide")
st.title("Premier League Predictions vs Actual Results")

# --------------------------------------------------
# Load data
# --------------------------------------------------

results_df = pd.read_csv("pl_results_from_api.csv")      # Finished matches
fixtures_df = pd.read_csv("pl_fixtures_from_api.csv")    # Upcoming fixtures
pred_df = pd.read_csv("pl_predictions.csv")              # Model predictions

# --------------------------------------------------
# Convert hybrid predictions to pred_result
# --------------------------------------------------
if "predicted_label" in pred_df.columns:
    pred_df["pred_result"] = pred_df["predicted_label"]
elif "pred_result" not in pred_df.columns:
    st.error("Your pl_predictions.csv needs either 'predicted_label' or 'pred_result'.")
    st.stop()

# --------------------------------------------------
# Add probability columns if they exist
# --------------------------------------------------
pred_merge_cols = [
    "season", "gameweek", "date", "home_team", "away_team", "pred_result"
]

for col in ["prob_H", "prob_D", "prob_A"]:
    if col in pred_df.columns and col not in pred_merge_cols:
        pred_merge_cols.append(col)

# --------------------------------------------------
# Merge predictions into RESULTS (played)
# --------------------------------------------------
played = results_df.merge(
    pred_df[pred_merge_cols],
    on=["season", "gameweek", "date", "home_team", "away_team"],
    how="left",
)

played["pred_result"] = played["pred_result"].fillna("")
played["has_pred"] = played["pred_result"].astype(str).str.strip() != ""
played["has_actual"] = played["actual_result"].notna()

played["correct"] = (
    played["has_pred"] & (played["pred_result"] == played["actual_result"])
)

played["correct_mark"] = np.where(
    played["has_pred"],
    played["correct"].map({True: "✅", False: "❌"}),
    "",
)

# --------------------------------------------------
# Merge predictions into FIXTURES (upcoming)
# --------------------------------------------------
upcoming = fixtures_df.merge(
    pred_df[pred_merge_cols],
    on=["season", "gameweek", "date", "home_team", "away_team"],
    how="left",
)

upcoming["pred_result"] = upcoming["pred_result"].fillna("")
upcoming["has_pred"] = upcoming["pred_result"].astype(str).str.strip() != ""
upcoming["actual_result"] = ""
upcoming["correct"] = False
upcoming["correct_mark"] = ""

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
view_mode = st.sidebar.radio("View", ["Played matches", "Upcoming fixtures"])

seasons = sorted(set(played["season"].unique()) | set(upcoming["season"].unique()))
selected_season = st.sidebar.selectbox("Season", seasons)

# --------------------------------------------------
# PLAYED MATCHES VIEW
# --------------------------------------------------
if view_mode == "Played matches":
    season_df = played[played["season"] == selected_season].copy()
    gw_list = sorted(season_df["gameweek"].dropna().unique())
    selected_gw = st.sidebar.selectbox("Gameweek", gw_list)

    gw_df = season_df[season_df["gameweek"] == selected_gw].copy()

    # Summary
    gw_matches_total = len(gw_df)
    gw_with_pred = gw_df["has_pred"].sum()
    gw_correct = gw_df["correct"].sum()
    gw_accuracy = (gw_correct / gw_with_pred * 100) if gw_with_pred > 0 else 0.0

    season_with_pred = season_df["has_pred"].sum()
    season_correct = season_df["correct"].sum()
    season_accuracy = (season_correct / season_with_pred * 100) if season_with_pred > 0 else 0.0

    st.subheader(f"Season {selected_season} · Gameweek {int(selected_gw)} · Played")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GW Matches (Played)", int(gw_matches_total))
    col2.metric("GW Predicted", int(gw_with_pred))
    col3.metric("GW Accuracy", f"{gw_accuracy:.1f}%")
    col4.metric("Season Accuracy", f"{season_accuracy:.1f}%")

    # GW Table (played)
    st.write("### Match Details (Selected Gameweek – Played)")

    gw_df = gw_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    gw_df["Row"] = gw_df.index + 1

    cols = ["Row", "date", "gameweek", "home_team", "away_team",
            "pred_result", "actual_result", "correct_mark"]

    gw_table = gw_df[cols].rename(columns={
        "Row": "#",
        "date": "Date",
        "gameweek": "GW",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "pred_result": "Your Prediction",
        "actual_result": "Actual Result",
        "correct_mark": "Correct?",
    })

    st.dataframe(gw_table, hide_index=True, use_container_width=True)

    # Season Table
    st.write("### All Played Matches This Season")
    season_df_sorted = season_df.sort_values(
        ["gameweek", "date", "home_team", "away_team"]
    ).reset_index(drop=True)
    season_df_sorted["Row"] = season_df_sorted.index + 1

    all_table = season_df_sorted[cols].rename(columns={
        "Row": "#",
        "date": "Date",
        "gameweek": "GW",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "pred_result": "Your Prediction",
        "actual_result": "Actual Result",
        "correct_mark": "Correct?",
    })

    st.dataframe(all_table, hide_index=True, use_container_width=True)

# --------------------------------------------------
# UPCOMING FIXTURES VIEW
# --------------------------------------------------
else:
    season_df = upcoming[upcoming["season"] == selected_season].copy()
    gw_list = sorted(season_df["gameweek"].dropna().unique())

    played_season = played[played["season"] == selected_season]
    if not played_season.empty:
        max_played_gw = played_season["gameweek"].max()
        future_gws = [gw for gw in gw_list if gw > max_played_gw]
        this_week_gw = min(future_gws) if future_gws else gw_list[0]
    else:
        this_week_gw = gw_list[0]

    default_index = gw_list.index(this_week_gw)
    selected_gw = st.sidebar.selectbox("Gameweek", gw_list, index=default_index)

    gw_df = season_df[season_df["gameweek"] == selected_gw].copy()

    st.subheader(f"Season {selected_season} · Gameweek {int(selected_gw)} · Upcoming")

    col1, col2 = st.columns(2)
    col1.metric("GW Fixtures", len(gw_df))
    col2.metric("GW Predicted", gw_df["has_pred"].sum())

    st.write("### Upcoming Fixtures (Selected Gameweek)")

    gw_df = gw_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    gw_df["Row"] = gw_df.index + 1

    # Add prob columns if available
    gw_display_cols = ["Row", "date", "gameweek",
                       "home_team", "away_team", "pred_result"]

    for col in ["prob_H", "prob_D", "prob_A"]:
        if col in gw_df.columns:
            gw_display_cols.append(col)

    gw_table = gw_df[gw_display_cols].rename(columns={
        "Row": "#",
        "date": "Date",
        "gameweek": "GW",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "pred_result": "Your Prediction",
        "prob_H": "P(Home)",
        "prob_D": "P(Draw)",
        "prob_A": "P(Away)",
    })

    st.dataframe(gw_table, hide_index=True, use_container_width=True)

    # Manual prediction editor (unchanged)
    st.write("### Add / Update Your Prediction for a Fixture in This Gameweek")

    if not gw_df.empty:
        gw_simple = gw_df.reset_index(drop=True)
        match_labels = [
            f"{row.home_team} vs {row.away_team} ({row.date})"
            for _, row in gw_simple.iterrows()
        ]
        selected_match_label = st.selectbox("Choose a match", match_labels)
        match_idx = match_labels.index(selected_match_label)
        match_row = gw_simple.iloc[match_idx]

        current_pred = str(match_row["pred_result"]).strip()
        options = ["H - Home win", "D - Draw", "A - Away win"]
        option_map = {"H": 0, "D": 1, "A": 2}
        default_opt = option_map.get(current_pred, 0)

        pred_choice_label = st.radio(
            "Your prediction",
            options,
            index=default_opt,
            horizontal=True,
        )

        pred_choice = pred_choice_label[0]

        if st.button("Save prediction"):
            pred_df_live = pd.read_csv("pl_predictions.csv")

            key_mask = (
                (pred_df_live["season"] == match_row["season"])
                & (pred_df_live["gameweek"] == match_row["gameweek"])
                & (pred_df_live["date"] == str(match_row["date"]))
                & (pred_df_live["home_team"] == match_row["home_team"])
                & (pred_df_live["away_team"] == match_row["away_team"])
            )

            if key_mask.any():
                pred_df_live.loc[key_mask, "pred_result"] = pred_choice
            else:
                new_row = {
                    "season": match_row["season"],
                    "gameweek": match_row["gameweek"],
                    "date": match_row["date"],
                    "home_team": match_row["home_team"],
                    "away_team": match_row["away_team"],
                    "pred_result": pred_choice,
                    "actual_result": "",
                }
                pred_df_live = pd.concat([pred_df_live, pd.DataFrame([new_row])],
                                         ignore_index=True)

            pred_df_live.to_csv("pl_predictions.csv", index=False)
            st.success(f"Saved prediction {pred_choice} for {selected_match_label}.")

    # All upcoming fixtures table
    st.write("### All Upcoming Fixtures This Season")

    season_df_sorted = season_df.sort_values(
        ["gameweek", "date", "home_team", "away_team"]
    ).reset_index(drop=True)
    season_df_sorted["Row"] = season_df_sorted.index + 1

    all_display_cols = [
        "Row",
        "date",
        "gameweek",
        "home_team",
        "away_team",
        "status",
        "pred_result",
    ]

    for col in ["prob_H", "prob_D", "prob_A"]:
        if col in season_df_sorted.columns:
            all_display_cols.append(col)

    all_table = season_df_sorted[all_display_cols].rename(columns={
        "Row": "#",
        "date": "Date",
        "gameweek": "GW",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "status": "Status",
        "pred_result": "Prediction",
        "prob_H": "P(Home)",
        "prob_D": "P(Draw)",
        "prob_A": "P(Away)",
    })

    st.dataframe(all_table, hide_index=True, use_container_width=True)
