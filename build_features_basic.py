import pandas as pd
import os

# -------------------------------------------------------------
# 1. Load matches_master.csv
# -------------------------------------------------------------
INPUT_PATH = "data/processed/matches_master.csv"
OUTPUT_PATH = "data/processed/match_features_basic.csv"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"{INPUT_PATH} not found. Run data_prep.py first.")

matches = pd.read_csv(INPUT_PATH)

# Ensure date type if needed
matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

# -------------------------------------------------------------
# 2. Convert match-level rows into team-level rows (long format)
# -------------------------------------------------------------
rows = []

for _, row in matches.iterrows():
    # Home team row
    rows.append({
        "match_id": row["match_id"],
        "team": row["home_team"],
        "opponent": row["away_team"],
        "is_home": 1,
        "goals_for": row["home_goals"],
        "goals_against": row["away_goals"],
        "result": row["result"],
        "season": row["season"],
        "gameweek": row["gameweek"],
        "date": row["date"],
    })

    # Away team row
    rows.append({
        "match_id": row["match_id"],
        "team": row["away_team"],
        "opponent": row["home_team"],
        "is_home": 0,
        "goals_for": row["away_goals"],
        "goals_against": row["home_goals"],
        "result": row["result"],
        "season": row["season"],
        "gameweek": row["gameweek"],
        "date": row["date"],
    })

team_df = pd.DataFrame(rows)

# -------------------------------------------------------------
# 3. Set numeric results (points)
# -------------------------------------------------------------
def result_to_points(team, opponent, goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    return 0

team_df["points"] = team_df.apply(
    lambda r: result_to_points(r["team"], r["opponent"], r["goals_for"], r["goals_against"]),
    axis=1
)

# Sort properly so rolling windows work
team_df = team_df.sort_values(["team", "date"])

# -------------------------------------------------------------
# 4. Rolling form features (LAST 5 MATCHES) — FIXED VERSION
# -------------------------------------------------------------
window = 5
g = team_df.groupby("team")

team_df["avg_goals_for_last5"] = g["goals_for"].transform(
    lambda s: s.shift().rolling(window, min_periods=1).mean()
)
team_df["avg_goals_against_last5"] = g["goals_against"].transform(
    lambda s: s.shift().rolling(window, min_periods=1).mean()
)
team_df["avg_points_last5"] = g["points"].transform(
    lambda s: s.shift().rolling(window, min_periods=1).mean()
)

# -------------------------------------------------------------
# 5. Merge team-level features back to match-level rows
# -------------------------------------------------------------
# Separate home/away features
home_features = team_df[team_df["is_home"] == 1].copy()
away_features = team_df[team_df["is_home"] == 0].copy()

home_features = home_features.rename(
    columns={
        "avg_goals_for_last5": "home_avg_goals_for_last5",
        "avg_goals_against_last5": "home_avg_goals_against_last5",
        "avg_points_last5": "home_avg_points_last5",
    }
)

away_features = away_features.rename(
    columns={
        "avg_goals_for_last5": "away_avg_goals_for_last5",
        "avg_goals_against_last5": "away_avg_goals_against_last5",
        "avg_points_last5": "away_avg_points_last5",
    }
)

# Merge onto matches
features = matches.merge(
    home_features[[
        "match_id", "team",
        "home_avg_goals_for_last5", "home_avg_goals_against_last5",
        "home_avg_points_last5"
    ]],
    left_on=["match_id", "home_team"],
    right_on=["match_id", "team"],
    how="left"
)

features = features.merge(
    away_features[[
        "match_id", "team",
        "away_avg_goals_for_last5", "away_avg_goals_against_last5",
        "away_avg_points_last5"
    ]],
    left_on=["match_id", "away_team"],
    right_on=["match_id", "team"],
    how="left"
)

# Drop extra team columns
features = features.drop(columns=["team_x", "team_y"], errors="ignore")

# -------------------------------------------------------------
# 6. Set ML target: home win / draw / away win
# -------------------------------------------------------------
features["target"] = features["result"]  # already 'H', 'D', 'A'

# -------------------------------------------------------------
# 7. Save final file
# -------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

features.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(features)} rows → {OUTPUT_PATH}")
