import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

MATCHES_PATH = PROCESSED_DIR / "matches_master.csv"
TRAIN_SUPER_PATH = PROCESSED_DIR / "match_features_super.csv"
FIXTURES_RAW_PATH = Path("pl_fixtures_from_api.csv")

OUT_PATH = PROCESSED_DIR / "pl_fixtures_features.csv"

# Elo params – same as in your build_elo_features.py
K = 20
HOME_ADVANTAGE = 100
BASE_RATING = 1500.0


def load_data():
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"{MATCHES_PATH} not found. Run data_prep.py + build_historical_matches.py first.")

    if not FIXTURES_RAW_PATH.exists():
        raise FileNotFoundError(f"{FIXTURES_RAW_PATH} not found. Run update_results.py first.")

    if not TRAIN_SUPER_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_SUPER_PATH} not found. Run build_super_features.py first.")

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    fixtures = pd.read_csv(FIXTURES_RAW_PATH, parse_dates=["date"])
    train_super = pd.read_csv(TRAIN_SUPER_PATH)

    return matches, fixtures, train_super


# -------------------------------------------------------------------
# Form features: last 5 games for each team before the fixture date
# -------------------------------------------------------------------
def compute_team_form(matches: pd.DataFrame, team: str, cutoff_date, window: int = 5):
    """Return (avg_gf, avg_ga, avg_pts) over last `window` games before cutoff_date."""
    team_games = matches[
        (matches["date"] < cutoff_date)
        & (
            (matches["home_team"] == team)
            | (matches["away_team"] == team)
        )
    ].sort_values("date")

    if team_games.empty:
        return 0.0, 0.0, 0.0

    last = team_games.tail(window)

    goals_for = []
    goals_against = []
    points = []

    for _, row in last.iterrows():
        res = row["result"]
        if row["home_team"] == team:
            gf = row["home_goals"]
            ga = row["away_goals"]
            if res == "H":
                pts = 3
            elif res == "D":
                pts = 1
            else:
                pts = 0
        else:  # team is away
            gf = row["away_goals"]
            ga = row["home_goals"]
            if res == "A":
                pts = 3
            elif res == "D":
                pts = 1
            else:
                pts = 0

        goals_for.append(gf)
        goals_against.append(ga)
        points.append(pts)

    n = len(last)
    return (
        float(np.sum(goals_for)) / n,
        float(np.sum(goals_against)) / n,
        float(np.sum(points)) / n,
    )


# -------------------------------------------------------------------
# Elo history from all past matches
# -------------------------------------------------------------------
def build_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Run Elo through historical matches and record rating AFTER each match for both teams."""
    ratings = {}
    records = []

    matches_sorted = matches.sort_values("date")

    for _, row in matches_sorted.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        res = row["result"]

        Rh = ratings.get(home, BASE_RATING)
        Ra = ratings.get(away, BASE_RATING)

        # Expected scores with home advantage
        Rh_eff = Rh + HOME_ADVANTAGE
        Ra_eff = Ra

        exp_home = 1.0 / (1.0 + 10 ** ((Ra_eff - Rh_eff) / 400.0))
        exp_away = 1.0 - exp_home

        if res == "H":
            Sh, Sa = 1.0, 0.0
        elif res == "A":
            Sh, Sa = 0.0, 1.0
        else:  # Draw
            Sh, Sa = 0.5, 0.5

        Rh_new = Rh + K * (Sh - exp_home)
        Ra_new = Ra + K * (Sa - exp_away)

        ratings[home] = Rh_new
        ratings[away] = Ra_new

        records.append({"team": home, "date": row["date"], "elo_after": Rh_new})
        records.append({"team": away, "date": row["date"], "elo_after": Ra_new})

    elo_hist = pd.DataFrame(records)
    return elo_hist


def get_team_elo(elo_hist: pd.DataFrame, team: str, cutoff_date) -> float:
    """Return the latest Elo AFTER the last match before cutoff_date."""
    if elo_hist.empty:
        return BASE_RATING

    hist = elo_hist[(elo_hist["team"] == team) & (elo_hist["date"] < cutoff_date)]
    if hist.empty:
        return BASE_RATING

    hist = hist.sort_values("date")
    return float(hist.iloc[-1]["elo_after"])


# -------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------
def build_fixture_super_features():
    matches, fixtures, train_super = load_data()

    # Build Elo history from all past matches
    print("Building Elo history from matches_master.csv ...")
    elo_hist = build_elo_history(matches)

    # Pre-compute means of odds/probability columns from training super features
    odds_prob_cols = [
        "home_odds_close",
        "draw_odds_close",
        "away_odds_close",
        "home_odds_b365",
        "draw_odds_b365",
        "away_odds_b365",
        "home_odds",
        "draw_odds",
        "away_odds",
        "prob_home_close",
        "prob_draw_close",
        "prob_away_close",
        "prob_home_b365",
        "prob_draw_b365",
        "prob_away_b365",
        "prob_home_market",
        "prob_draw_market",
        "prob_away_market",
        "prob_home_raw",
        "prob_draw_raw",
        "prob_away_raw",
        "odds_imbalance_home_away",
    ]

    odds_means = {}
    for col in odds_prob_cols:
        if col in train_super.columns:
            odds_means[col] = train_super[col].mean(skipna=True)
        else:
            odds_means[col] = np.nan

    rows = []

    print(f"Building features for {len(fixtures)} upcoming fixtures...")
    for _, row in fixtures.iterrows():
        season = row["season"]
        gw = row["gameweek"]
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        status = row.get("status", "")

        # Form features
        (
            home_gf,
            home_ga,
            home_pts,
        ) = compute_team_form(matches, home, date)
        (
            away_gf,
            away_ga,
            away_pts,
        ) = compute_team_form(matches, away, date)

        # Elo features
        home_elo = get_team_elo(elo_hist, home, date)
        away_elo = get_team_elo(elo_hist, away, date)
        elo_diff = home_elo - away_elo
        denom = home_elo + away_elo
        if denom <= 0:
            elo_ratio_home = 0.5
            elo_ratio_away = 0.5
        else:
            elo_ratio_home = home_elo / denom
            elo_ratio_away = away_elo / denom

        record = {
            "season": season,
            "gameweek": gw,
            "date": date,
            "home_team": home,
            "away_team": away,
            "status": status,
            # Form
            "home_avg_goals_for_last5": home_gf,
            "home_avg_goals_against_last5": home_ga,
            "home_avg_points_last5": home_pts,
            "away_avg_goals_for_last5": away_gf,
            "away_avg_goals_against_last5": away_ga,
            "away_avg_points_last5": away_pts,
            # Elo
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "elo_diff": elo_diff,
            "elo_ratio_home": elo_ratio_home,
            "elo_ratio_away": elo_ratio_away,
        }

        # Odds + probability features – we don't know them for future,
        # so we plug in the historical mean as a neutral baseline.
        for col, mean_val in odds_means.items():
            record[col] = mean_val

        rows.append(record)

    feat_df = pd.DataFrame(rows)

    # Ensure columns order roughly matches training super features (not strictly required)
    base_cols = [
        "season",
        "gameweek",
        "date",
        "home_team",
        "away_team",
        "status",
    ]
    feature_cols = [
        "home_avg_goals_for_last5",
        "home_avg_goals_against_last5",
        "home_avg_points_last5",
        "away_avg_goals_for_last5",
        "away_avg_goals_against_last5",
        "away_avg_points_last5",
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
        "elo_ratio_home",
        "elo_ratio_away",
    ] + odds_prob_cols

    ordered_cols = base_cols + feature_cols
    feat_df = feat_df[ordered_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(OUT_PATH, index=False)
    print(f"Saved fixture features to {OUT_PATH}")
    print(f"Rows: {len(feat_df)}")
    print("Columns:", list(feat_df.columns))


if __name__ == "__main__":
    build_fixture_super_features()
