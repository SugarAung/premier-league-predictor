import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_START_GW = 12            # model goes live from this GW onwards
CURRENT_SEASON = "2025-2026"   # only care about this season

# Base paths (project root = folder that contains /app, /data, /models, /scripts)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LIVE_DIR = DATA_DIR / "live"
PROCESSED_DIR = DATA_DIR / "processed"

PRED_PATH = LIVE_DIR / "pl_predictions.csv"             # live predictions
RESULTS_PATH = LIVE_DIR / "pl_results_from_api.csv"     # finished matches from API
FIXTURES_PATH = LIVE_DIR / "pl_fixtures_from_api.csv"   # upcoming fixtures from API
BACKTEST_PATH = PROCESSED_DIR / "backtest_hybrid.csv"   # backtest output

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Premier League Predictions", layout="wide")

# Simple title (no logo / no gap)
st.title("Premier League Predictions vs Actual Results")
st.caption("Model: XGBoost Hybrid (75% sharp odds model, 25% full-history model)")

st.markdown(
    """
    Use the controls on the left to pick **Backtest** or **Live model**.

    - **Backtest** – shows how the current model would have performed on finished
      gameweeks **before it went live** (e.g. GW 1–11).
    - **Live model** – shows **this week’s predictions**, **past live weeks**, and
      **upcoming fixtures** from the moment the model went live (GW ≥ 12).
    """
)

# --------------------------------------------------
# File checks for live data
# --------------------------------------------------
missing_files = [p for p in [RESULTS_PATH, FIXTURES_PATH] if not p.exists()]
if missing_files:
    st.error(
        "The following required data file(s) are missing:\n\n"
        + "\n".join(f"- `{f.relative_to(ROOT_DIR)}`" for f in missing_files)
        + "\n\nPlease make sure they exist in the `data/live` folder."
    )
    st.stop()

if PRED_PATH.exists():
    mtime = datetime.fromtimestamp(PRED_PATH.stat().st_mtime)
    st.caption(f"Live predictions last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.info("`pl_predictions.csv` not found yet – live model views will show without model picks.")

# --------------------------------------------------
# Load LIVE results, fixtures, predictions
# --------------------------------------------------
results_df = pd.read_csv(RESULTS_PATH)     # Finished matches
fixtures_df = pd.read_csv(FIXTURES_PATH)   # Upcoming fixtures

if PRED_PATH.exists():
    live_pred_df = pd.read_csv(PRED_PATH)
else:
    live_pred_df = pd.DataFrame(
        columns=["season", "gameweek", "date", "home_team", "away_team", "pred_result"]
    )

# Ensure pred_result column exists
if "predicted_label" in live_pred_df.columns:
    live_pred_df["pred_result"] = live_pred_df["predicted_label"]
elif "pred_result" not in live_pred_df.columns:
    live_pred_df["pred_result"] = ""

# Drop duplicate predictions for the same match (keep last)
match_key = ["season", "gameweek", "date", "home_team", "away_team"]
if all(c in live_pred_df.columns for c in match_key):
    live_pred_df = live_pred_df.drop_duplicates(subset=match_key, keep="last")

# Build list of prediction columns to merge
pred_merge_cols = ["season", "gameweek", "date", "home_team", "away_team", "pred_result"]
for col in ["prob_H", "prob_D", "prob_A"]:
    if col in live_pred_df.columns and col not in pred_merge_cols:
        pred_merge_cols.append(col)

# Merge into RESULTS (played)
played = results_df.merge(
    live_pred_df[pred_merge_cols],
    on=["season", "gameweek", "date", "home_team", "away_team"],
    how="left",
)

played["pred_result"] = played["pred_result"].fillna("")
played["has_pred"] = played["pred_result"].astype(str).str.strip() != ""
played["has_actual"] = played["actual_result"].notna()
played["correct"] = played["has_pred"] & (played["pred_result"] == played["actual_result"])
played["correct_mark"] = np.where(
    played["has_pred"],
    played["correct"].map({True: "✅", False: "❌"}),
    "",
)

if {"prob_H", "prob_D", "prob_A"}.issubset(played.columns):
    played["model_confidence"] = played[["prob_H", "prob_D", "prob_A"]].max(axis=1)
else:
    played["model_confidence"] = np.nan

# Merge into FIXTURES (upcoming)
upcoming = fixtures_df.merge(
    live_pred_df[pred_merge_cols],
    on=["season", "gameweek", "date", "home_team", "away_team"],
    how="left",
)

upcoming["pred_result"] = upcoming["pred_result"].fillna("")
upcoming["has_pred"] = upcoming["pred_result"].astype(str).str.strip() != ""
upcoming["actual_result"] = ""
upcoming["correct"] = False
upcoming["correct_mark"] = ""

if {"prob_H", "prob_D", "prob_A"}.issubset(upcoming.columns):
    upcoming["model_confidence"] = upcoming[["prob_H", "prob_D", "prob_A"]].max(axis=1)
else:
    upcoming["model_confidence"] = np.nan

# --------------------------------------------------
# Load BACKTEST data (from backtest_hybrid.py)
# --------------------------------------------------
if BACKTEST_PATH.exists():
    backtest_df = pd.read_csv(BACKTEST_PATH)

    if "predicted_label" in backtest_df.columns:
        backtest_df["pred_result"] = backtest_df["predicted_label"]
    else:
        backtest_df["pred_result"] = backtest_df.get("pred_result", "")

    if "actual_result" not in backtest_df.columns and "result" in backtest_df.columns:
        backtest_df["actual_result"] = backtest_df["result"]

    if "correct" not in backtest_df.columns and \
       {"pred_result", "actual_result"}.issubset(backtest_df.columns):
        backtest_df["correct"] = backtest_df["pred_result"] == backtest_df["actual_result"]

    # >>> FIXED: only treat real, non-empty values as predictions
    backtest_df["has_pred"] = backtest_df["pred_result"].notna() & (
        backtest_df["pred_result"].astype(str).str.strip() != ""
    )

    backtest_df["correct_mark"] = np.where(
        backtest_df["has_pred"] & backtest_df["correct"],
        "✅",
        np.where(backtest_df["has_pred"] & ~backtest_df["correct"], "❌", ""),
    )

    if {"prob_H", "prob_D", "prob_A"}.issubset(backtest_df.columns):
        backtest_df["model_confidence"] = backtest_df[["prob_H", "prob_D", "prob_A"]].max(axis=1)
    else:
        backtest_df["model_confidence"] = np.nan
else:
    backtest_df = None

# --------------------------------------------------
# Sidebar controls (no season dropdown)
# --------------------------------------------------
st.sidebar.header("Controls")

selected_season = CURRENT_SEASON  # fixed, no dropdown

page_mode = st.sidebar.radio(
    "Page",
    ["Backtest (before live)", f"Live model (GW ≥ {MODEL_START_GW})"],
)

# Filter by season
season_played = played[played["season"] == selected_season].copy()
season_upcoming = upcoming[upcoming["season"] == selected_season].copy()
if backtest_df is not None and "season" in backtest_df.columns:
    season_backtest = backtest_df[backtest_df["season"] == selected_season].copy()
else:
    season_backtest = pd.DataFrame()

# Split live played by GW (for live page)
played_live = season_played[season_played["gameweek"] >= MODEL_START_GW].copy()
upcoming_live = season_upcoming[season_upcoming["gameweek"] >= MODEL_START_GW].copy()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def accuracy_for_df(df: pd.DataFrame) -> float:
    if "correct" not in df.columns or "has_pred" not in df.columns:
        return 0.0
    with_pred = df["has_pred"].sum()
    correct = df["correct"].sum()
    return (correct / with_pred * 100.0) if with_pred > 0 else 0.0

def accuracy_up_to(df: pd.DataFrame, gw_limit: int) -> float:
    subset = df[df["gameweek"] <= gw_limit]
    return accuracy_for_df(subset)

# ==================================================
#  BACKTEST PAGE
# ==================================================
if page_mode.startswith("Backtest"):
    st.subheader(f"Backtest · Season {selected_season} · Gameweeks before {MODEL_START_GW}")

    if backtest_df is None or season_backtest.empty:
        st.info(
            "No backtest file found or no backtest data for this season.\n\n"
            "Generate one with something like:\n\n"
            "`python scripts/backtest_hybrid.py --min_season 2025-2026 "
            "--output data/processed/backtest_hybrid.csv`"
        )
    else:
        # Only include pre-live gameweeks
        season_backtest = season_backtest[season_backtest["gameweek"] < MODEL_START_GW].copy()
        # and only rows where we actually have a prediction
        season_backtest = season_backtest[season_backtest["has_pred"]].copy()

        if season_backtest.empty:
            st.info(f"No pre-live gameweeks (GW < {MODEL_START_GW}) with predictions in backtest data for this season.")
        else:
            gw_list_bt = sorted(season_backtest["gameweek"].unique())
            default_idx = len(gw_list_bt) - 1
            selected_bt_gw = st.selectbox(
                "Select backtest gameweek",
                gw_list_bt,
                index=default_idx,
            )
            gw_bt = int(selected_bt_gw)

            gw_df = season_backtest[season_backtest["gameweek"] == gw_bt].copy()

            gw_acc = accuracy_for_df(gw_df)
            overall_bt_acc = accuracy_for_df(season_backtest)

            if "model_confidence" in gw_df.columns and gw_df["model_confidence"].notna().any():
                gw_conf = gw_df["model_confidence"].mean() * 100
            else:
                gw_conf = np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("Backtest GW", f"{gw_bt}")
            c2.metric("GW Backtest Accuracy", f"{gw_acc:.1f}%")
            c3.metric("Overall Backtest Accuracy", f"{overall_bt_acc:.1f}%")

            st.caption(
                f"Mode: **Backtest** – predictions computed on finished matches for "
                f"GW < {MODEL_START_GW}, to see how the model would have performed "
                "before going live."
            )

            # Table for selected backtest GW (only rows with predictions)
            st.write("### Backtest Match Details (Selected Gameweek)")
            gw_df = gw_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
            gw_df["Row"] = gw_df.index + 1

            cols = [
                "Row", "date", "gameweek", "home_team", "away_team",
                "pred_result", "actual_result", "correct_mark", "model_confidence"
            ]
            cols = [c for c in cols if c in gw_df.columns]

            table = gw_df[cols].rename(columns={
                "Row": "#",
                "date": "Date",
                "gameweek": "GW",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "pred_result": "Model Pick",
                "actual_result": "Actual Result",
                "correct_mark": "Correct?",
                "model_confidence": "Confidence",
            })

            if "Confidence" in table.columns:
                table["Confidence"] = table["Confidence"].map(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
                )

            st.dataframe(table, hide_index=True, use_container_width=True)

            # All backtest matches (only rows with predictions)
            with st.expander("Show all backtest matches for this season"):
                all_bt = season_backtest.sort_values(
                    ["gameweek", "date", "home_team", "away_team"]
                ).reset_index(drop=True)
                all_bt["Row"] = all_bt.index + 1

                all_cols = [
                    "Row", "date", "gameweek", "home_team", "away_team",
                    "pred_result", "actual_result", "correct_mark", "model_confidence"
                ]
                # FIX: respect desired order
                all_cols = [c for c in all_cols if c in all_bt.columns]

                all_table = all_bt[all_cols].rename(columns={
                    "Row": "#",
                    "date": "Date",
                    "gameweek": "GW",
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "pred_result": "Model Pick",
                    "actual_result": "Actual Result",
                    "correct_mark": "Correct?",
                    "model_confidence": "Confidence",
                })

                if "Confidence" in all_table.columns:
                    all_table["Confidence"] = all_table["Confidence"].map(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
                    )

                st.dataframe(all_table, hide_index=True, use_container_width=True)

# ==================================================
#  LIVE MODEL PAGE
# ==================================================
else:
    st.subheader(f"Live Model · Season {selected_season} · Gameweeks ≥ {MODEL_START_GW}")

    # ---------- THIS WEEK PREDICTIONS ----------
    if upcoming_live.empty:
        st.info("No upcoming fixtures in the live model period for this season.")
    else:
        # Determine "this week" as the smallest GW > last played live GW,
        # or smallest upcoming live GW if none played yet.
        if played_live.empty:
            this_week_gw = int(upcoming_live["gameweek"].min())
        else:
            max_played_live_gw = int(played_live["gameweek"].max())
            future_gws = sorted(
                gw for gw in upcoming_live["gameweek"].unique()
                if gw > max_played_live_gw
            )
            this_week_gw = int(future_gws[0]) if future_gws else int(upcoming_live["gameweek"].min())

        gw_this = upcoming_live[upcoming_live["gameweek"] == this_week_gw].copy()
        gw_matches_total = len(gw_this)
        gw_with_pred = gw_this["has_pred"].sum()
        gw_conf = (
            gw_this["model_confidence"].mean() * 100
            if "model_confidence" in gw_this.columns and gw_this["model_confidence"].notna().any()
            else np.nan
        )

        live_overall_acc = accuracy_for_df(played_live) if not played_live.empty else 0.0

        st.markdown(f"### 🔮 This Week Predictions · Gameweek {this_week_gw}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GW Fixtures", int(gw_matches_total))
        col2.metric("GW Predicted", int(gw_with_pred))
        col3.metric("Avg GW Confidence", f"{gw_conf:.1f}%" if not np.isnan(gw_conf) else "N/A")
        col4.metric(f"Overall Live Accuracy (GW ≥ {MODEL_START_GW})", f"{live_overall_acc:.1f}%")

        if not gw_this.empty:
            gw_this = gw_this.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
            gw_this["Row"] = gw_this.index + 1

            cols = [
                "Row", "date", "gameweek", "home_team", "away_team",
                "pred_result", "prob_H", "prob_D", "prob_A", "model_confidence"
            ]
            cols = [c for c in cols if c in gw_this.columns]

            table = gw_this[cols].rename(columns={
                "Row": "#",
                "date": "Date",
                "gameweek": "GW",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "pred_result": "Model Pick",
                "prob_H": "P(Home)",
                "prob_D": "P(Draw)",
                "prob_A": "P(Away)",
                "model_confidence": "Confidence",
            })

            for col in ["P(Home)", "P(Draw)", "P(Away)", "Confidence"]:
                if col in table.columns:
                    table[col] = table[col].map(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
                    )

            st.dataframe(table, hide_index=True, use_container_width=True)

    # ---------- PAST LIVE WEEKS ----------
    st.markdown("### ✅ Past Live Weeks (Finished Matches with Predictions)")

    if played_live.empty:
        st.info(f"No finished matches yet in live era (from GW {MODEL_START_GW}).")
    else:
        gw_list_live = sorted(played_live["gameweek"].unique())
        default_idx = len(gw_list_live) - 1
        selected_live_gw = st.selectbox(
            "Select live gameweek to review",
            gw_list_live,
            index=default_idx,
        )
        gw_live = int(selected_live_gw)
        gw_df = played_live[played_live["gameweek"] == gw_live].copy()

        gw_acc = accuracy_for_df(gw_df)
        live_acc_up_to = accuracy_up_to(played_live, gw_live)

        gw_conf_live = (
            gw_df["model_confidence"].mean() * 100
            if "model_confidence" in gw_df.columns and gw_df["model_confidence"].notna().any()
            else np.nan
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Live GW", f"{gw_live}")
        c2.metric("GW Accuracy (live)", f"{gw_acc:.1f}%")
        c3.metric("Live Accuracy up to this GW", f"{live_acc_up_to:.1f}%")

        gw_df = gw_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
        gw_df["Row"] = gw_df.index + 1

        cols = [
            "Row", "date", "gameweek", "home_team", "away_team",
            "pred_result", "actual_result", "correct_mark", "model_confidence"
        ]
        cols = [c for c in cols if c in gw_df.columns]

        table_live = gw_df[cols].rename(columns={
            "Row": "#",
            "date": "Date",
            "gameweek": "GW",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "pred_result": "Model Pick",
            "actual_result": "Actual Result",
            "correct_mark": "Correct?",
            "model_confidence": "Confidence",
        })

        if "Confidence" in table_live.columns:
            table_live["Confidence"] = table_live["Confidence"].map(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
            )

        st.dataframe(table_live, hide_index=True, use_container_width=True)

        # All finished live matches
        with st.expander("Show all finished matches in live era"):
            all_live = played_live.sort_values(
                ["gameweek", "date", "home_team", "away_team"]
            ).reset_index(drop=True)
            all_live["Row"] = all_live.index + 1

            all_cols = [
                "Row", "date", "gameweek", "home_team", "away_team",
                "pred_result", "actual_result", "correct_mark", "model_confidence"
            ]
            all_cols = [c for c in all_live.columns if c in all_cols]

            all_table_live = all_live[all_cols].rename(columns={
                "Row": "#",
                "date": "Date",
                "gameweek": "GW",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "pred_result": "Model Pick",
                "actual_result": "Actual Result",
                "correct_mark": "Correct?",
                "model_confidence": "Confidence",
            })

            if "Confidence" in all_table_live.columns:
                all_table_live["Confidence"] = all_table_live["Confidence"].map(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
                )

            st.dataframe(all_table_live, hide_index=True, use_container_width=True)

    # ---------- UPCOMING LIVE MATCHES ----------
    st.markdown("### 📅 All Upcoming Matches in Live Era")

    if upcoming_live.empty:
        st.info("No upcoming fixtures left in the live model period for this season.")
    else:
        up_all = upcoming_live.sort_values(
            ["gameweek", "date", "home_team", "away_team"]
        ).reset_index(drop=True)
        up_all["Row"] = up_all.index + 1

        cols = [
            "Row", "date", "gameweek", "home_team", "away_team",
            "status", "pred_result", "prob_H", "prob_D", "prob_A", "model_confidence"
        ]
        cols = [c for c in cols if c in up_all.columns]

        up_table = up_all[cols].rename(columns={
            "Row": "#",
            "date": "Date",
            "gameweek": "GW",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "status": "Status",
            "pred_result": "Model Pick",
            "prob_H": "P(Home)",
            "prob_D": "P(Draw)",
            "prob_A": "P(Away)",
            "model_confidence": "Confidence",
        })

        for col in ["P(Home)", "P(Draw)", "P(Away)", "Confidence"]:
            if col in up_table.columns:
                up_table[col] = up_table[col].map(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
                )

        st.dataframe(up_table, hide_index=True, use_container_width=True)
