import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MATCHES_PATH = PROCESSED_DIR / "matches_master.csv"
BASIC_FEATS_PATH = PROCESSED_DIR / "match_features_basic.csv"
OUT_PATH = PROCESSED_DIR / "match_features_with_elo.csv"

K = 20               # Elo K-factor
HOME_ADVANTAGE = 100 # Elo points given to home team


def build_elo_features():
    if not MATCHES_PATH.exists():
        raise FileNotFoundError("Run data_prep.py first.")
    if not BASIC_FEATS_PATH.exists():
        raise FileNotFoundError("Run build_features_basic.py first.")

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    matches = matches.sort_values(["season", "date"]).reset_index(drop=True)

    ratings = {}  # team -> current elo
    records = []  # list of dicts with elo_before for each match

    # helper to get current rating or default 1500
    def get_rating(team):
        return ratings.get(team, 1500.0)

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]  # 'H', 'D', 'A'

        Rh = get_rating(home)
        Ra = get_rating(away)

        # record elo BEFORE this match
        records.append(
            {
                "match_id": row["match_id"],
                "home_elo_before": Rh,
                "away_elo_before": Ra,
            }
        )

        # expected scores with home advantage
        Rh_eff = Rh + HOME_ADVANTAGE
        Ra_eff = Ra

        Eh = 1 / (1 + 10 ** ((Ra_eff - Rh_eff) / 400))
        Ea = 1 - Eh

        if result == "H":
            Sh, Sa = 1.0, 0.0
        elif result == "A":
            Sh, Sa = 0.0, 1.0
        else:  # draw
            Sh, Sa = 0.5, 0.5

        # update ratings
        Rh_new = Rh + K * (Sh - Eh)
        Ra_new = Ra + K * (Sa - Ea)
        ratings[home] = Rh_new
        ratings[away] = Ra_new

    elo_df = pd.DataFrame(records)

    # merge onto basic features
    basic = pd.read_csv(BASIC_FEATS_PATH, parse_dates=["date"])
    merged = basic.merge(elo_df, on="match_id", how="left")

    # useful extra feature
    merged["elo_diff"] = merged["home_elo_before"] - merged["away_elo_before"]

    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(merged)} rows to {OUT_PATH}")


if __name__ == "__main__":
    build_elo_features()
