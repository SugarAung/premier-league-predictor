import numpy as np
import pandas as pd
from pathlib import Path

# Where your raw season CSVs live
RAW_HIST_DIR = Path("data/raw/historical")
PROCESSED_DIR = Path("data/processed")

# Input: existing Elo features file
ELO_FEATURES_PATH = PROCESSED_DIR / "match_features_with_elo.csv"

# Output: richer feature file
OUT_PATH = PROCESSED_DIR / "match_features_super.csv"

# Same mapping as in build_historical_matches.py
HIST_SEASONS = [
    ("2014-2015", "E0_2014_2015.csv"),
    ("2015-2016", "E0_2015_2016.csv"),
    ("2016-2017", "E0_2016_2017.csv"),
    ("2017-2018", "E0_2017_2018.csv"),
    ("2018-2019", "E0_2018_2019.csv"),
    ("2019-2020", "E0_2019_2020.csv"),
    ("2020-2021", "E0_2020_2021.csv"),
    ("2021-2022", "E0_2021_2022.csv"),
    ("2022-2023", "E0_2022_2023.csv"),
    ("2023-2024", "E0_2023_2024.csv"),
]


def load_odds_for_season(season_str: str, filename: str) -> pd.DataFrame:
    """
    Load one raw E0_xxx CSV and extract odds + keys we need to join
    onto the Elo feature file.

    We use BOTH:
      - Closing Pinnacle odds (PSCH, PSCD, PSCA) when available
      - Bet365 odds (B365H, B365D, B365A) when available

    Then we build market probabilities using whatever is present
    (Option A: keep all matches, use best available odds).
    """
    path = RAW_HIST_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    df = pd.read_csv(path)

    odds = df[["Date", "HomeTeam", "AwayTeam"]].copy()
    cols = df.columns

    # 1) Closing Pinnacle odds
    if {"PSCH", "PSCD", "PSCA"}.issubset(cols):
        odds["home_odds_close"] = df["PSCH"]
        odds["draw_odds_close"] = df["PSCD"]
        odds["away_odds_close"] = df["PSCA"]
    else:
        odds["home_odds_close"] = np.nan
        odds["draw_odds_close"] = np.nan
        odds["away_odds_close"] = np.nan

    # 2) Bet365 odds
    if {"B365H", "B365D", "B365A"}.issubset(cols):
        odds["home_odds_b365"] = df["B365H"]
        odds["draw_odds_b365"] = df["B365D"]
        odds["away_odds_b365"] = df["B365A"]
    else:
        odds["home_odds_b365"] = np.nan
        odds["draw_odds_b365"] = np.nan
        odds["away_odds_b365"] = np.nan

    def add_implied_probs(df_odds: pd.DataFrame, prefix: str) -> pd.DataFrame:
        h = f"home_odds_{prefix}"
        d = f"draw_odds_{prefix}"
        a = f"away_odds_{prefix}"

        if not {h, d, a}.issubset(df_odds.columns):
            return df_odds

        inv = 1.0 / df_odds[[h, d, a]]
        inv.replace([np.inf, -np.inf], np.nan, inplace=True)
        s = inv.sum(axis=1)

        df_odds[f"prob_home_{prefix}"] = inv[h] / s
        df_odds[f"prob_draw_{prefix}"] = inv[d] / s
        df_odds[f"prob_away_{prefix}"] = inv[a] / s
        return df_odds

    odds = add_implied_probs(odds, "close")
    odds = add_implied_probs(odds, "b365")

    # Market average probabilities:
    # use whatever exists (skipna=True), so if only one book is present, we still get a value.
    odds["prob_home_market"] = odds[
        ["prob_home_close", "prob_home_b365"]
    ].mean(axis=1, skipna=True)
    odds["prob_draw_market"] = odds[
        ["prob_draw_close", "prob_draw_b365"]
    ].mean(axis=1, skipna=True)
    odds["prob_away_market"] = odds[
        ["prob_away_close", "prob_away_b365"]
    ].mean(axis=1, skipna=True)

    # Backwards-compatible single odds / raw probs, also using best-available view
    odds["home_odds"] = odds[["home_odds_close", "home_odds_b365"]].mean(
        axis=1, skipna=True
    )
    odds["draw_odds"] = odds[["draw_odds_close", "draw_odds_b365"]].mean(
        axis=1, skipna=True
    )
    odds["away_odds"] = odds[["away_odds_close", "away_odds_b365"]].mean(
        axis=1, skipna=True
    )

    odds["prob_home_raw"] = odds["prob_home_market"]
    odds["prob_draw_raw"] = odds["prob_draw_market"]
    odds["prob_away_raw"] = odds["prob_away_market"]

    # Keys to match Elo features
    odds = odds.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
        }
    )

    odds["date"] = pd.to_datetime(odds["date"], format="%d/%m/%y", errors="coerce")
    odds["season"] = season_str

    # Extra interaction feature
    odds["odds_imbalance_home_away"] = (
        odds["prob_home_market"] - odds["prob_away_market"]
    )

    return odds


def build_super_features():
    if not ELO_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{ELO_FEATURES_PATH} not found. Run build_elo_features.py first."
        )

    feat = pd.read_csv(ELO_FEATURES_PATH, parse_dates=["date"])

    odds_dfs = []
    for season_str, filename in HIST_SEASONS:
        print(f"Loading odds for season {season_str} from {filename}...")
        odds_df = load_odds_for_season(season_str, filename)
        odds_dfs.append(odds_df)

    odds_all = pd.concat(odds_dfs, ignore_index=True)

    merged = feat.merge(
        odds_all,
        on=["season", "date", "home_team", "away_team"],
        how="left",
        validate="m:1",
    )

    # Relative Elo strength
    denom = merged["home_elo_before"] + merged["away_elo_before"]
    merged["elo_ratio_home"] = merged["home_elo_before"] / denom
    merged["elo_ratio_away"] = merged["away_elo_before"] / denom

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved super feature file to {OUT_PATH}")
    print(f"Rows: {len(merged)}")
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    build_super_features()
