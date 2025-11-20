import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path("data/processed")
MATCHES_PATH = DATA_DIR / "matches_master.csv"
MODEL_PATH = Path("models/basic_form_model.pkl")

WINDOW = 5  # same rolling window as in build_features_basic.py


def load_matches() -> pd.DataFrame:
    """Load historical matches (finished games)."""
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHES_PATH} not found. Run data_prep.py and build_features_basic.py first."
        )
    df = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    return df


def build_team_rolling_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build team-level rolling stats table (same logic as build_features_basic.py),
    so we can get the latest form for each team.
    """
    rows = []
    for _, row in matches.iterrows():
        # Home
        rows.append(
            {
                "match_id": row["match_id"],
                "team": row["home_team"],
                "opponent": row["away_team"],
                "is_home": 1,
                "goals_for": row["home_goals"],
                "goals_against": row["away_goals"],
                "season": row["season"],
                "gameweek": row["gameweek"],
                "date": row["date"],
            }
        )
        # Away
        rows.append(
            {
                "match_id": row["match_id"],
                "team": row["away_team"],
                "opponent": row["home_team"],
                "is_home": 0,
                "goals_for": row["away_goals"],
                "goals_against": row["home_goals"],
                "season": row["season"],
                "gameweek": row["gameweek"],
                "date": row["date"],
            }
        )

    team_df = pd.DataFrame(rows)
    team_df = team_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

    # Compute points from each team's POV
    def gf_ga_to_points(gf, ga):
        if gf > ga:
            return 3
        elif gf == ga:
            return 1
        return 0

    team_df["points"] = team_df.apply(
        lambda r: gf_ga_to_points(r["goals_for"], r["goals_against"]), axis=1
    )

    # Rolling features (shift so we don't include the current match)
    g = team_df.groupby("team")

    team_df["avg_goals_for_last5"] = g["goals_for"].transform(
        lambda s: s.shift().rolling(WINDOW, min_periods=1).mean()
    )
    team_df["avg_goals_against_last5"] = g["goals_against"].transform(
        lambda s: s.shift().rolling(WINDOW, min_periods=1).mean()
    )
    team_df["avg_points_last5"] = g["points"].transform(
        lambda s: s.shift().rolling(WINDOW, min_periods=1).mean()
    )

    # For the first matches of each team there is no history -> NaN -> set to 0
    feature_cols = [
        "avg_goals_for_last5",
        "avg_goals_against_last5",
        "avg_points_last5",
    ]
    team_df[feature_cols] = team_df[feature_cols].fillna(0.0)

    return team_df


def get_latest_team_features(team_df: pd.DataFrame, team_name: str) -> Tuple[float, float, float]:
    """
    Get the most recent rolling stats for a given team.
    Returns (avg_goals_for_last5, avg_goals_against_last5, avg_points_last5).
    """
    team_rows = team_df[team_df["team"] == team_name]

    if team_rows.empty:
        raise ValueError(f"No matches found for team '{team_name}' in historical data.")

    last_row = team_rows.sort_values("date").iloc[-1]

    return (
        float(last_row["avg_goals_for_last5"]),
        float(last_row["avg_goals_against_last5"]),
        float(last_row["avg_points_last5"]),
    )


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_model_basic.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    model = obj["model"]
    feature_cols = obj["feature_cols"]
    return model, feature_cols


def predict_match(home_team: str, away_team: str) -> Dict[str, float]:
    """
    Predict probabilities for Home win / Draw / Away win
    based on current rolling form (last 5 matches) of both teams.
    """
    matches = load_matches()
    team_df = build_team_rolling_table(matches)

    # Get latest form stats for both teams
    (
        home_gf5,
        home_ga5,
        home_pts5,
    ) = get_latest_team_features(team_df, home_team)
    (
        away_gf5,
        away_ga5,
        away_pts5,
    ) = get_latest_team_features(team_df, away_team)

    model, feature_cols = load_model()

    # Build feature row in the correct order
    feature_row = {
        "home_avg_goals_for_last5": home_gf5,
        "home_avg_goals_against_last5": home_ga5,
        "home_avg_points_last5": home_pts5,
        "away_avg_goals_for_last5": away_gf5,
        "away_avg_goals_against_last5": away_ga5,
        "away_avg_points_last5": away_pts5,
    }

    X = np.array([[feature_row[col] for col in feature_cols]])

    probs = model.predict_proba(X)[0]  # array of probabilities
    classes = model.classes_            # e.g. ['A', 'D', 'H']

    # Map classes to friendly labels
    label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    result = {}
    for cls, p in zip(classes, probs):
        key = label_map.get(cls, cls)
        result[key] = float(p)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict result probabilities for a match using the basic form model."
    )
    parser.add_argument("home_team", type=str, help="Home team name exactly as in CSV")
    parser.add_argument("away_team", type=str, help="Away team name exactly as in CSV")

    args = parser.parse_args()

    probs = predict_match(args.home_team, args.away_team)

    print(f"\nPrediction based on current form (last {WINDOW} matches):")
    print(f"Home team: {args.home_team}")
    print(f"Away team: {args.away_team}\n")

    for outcome, p in probs.items():
        print(f"{outcome}: {p:.3f}")

    # Also show which outcome has highest probability
    best_outcome = max(probs.items(), key=lambda kv: kv[1])
    print(f"\nMost likely: {best_outcome[0]} ({best_outcome[1]:.3f})")


if __name__ == "__main__":
    main()
