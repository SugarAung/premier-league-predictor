import pandas as pd
from pathlib import Path

RAW_HIST_DIR = Path("data/raw/historical")
PROCESSED_DIR = Path("data/processed")
CURRENT_MATCHES_PATH = PROCESSED_DIR / "matches_master.csv"  # your 2025-2026 season
OUT_PATH = PROCESSED_DIR / "matches_master.csv"  # we'll overwrite with ALL seasons


# List of (season_string, filename) to load
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


def load_historical_season(season_str: str, filename: str) -> pd.DataFrame:
    """Load one Football-Data CSV and convert to our standard matches schema."""
    path = RAW_HIST_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Put it in data/raw/historical/")

    df = pd.read_csv(path)

    # Make sure the essential columns exist
    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    # Standardize columns
    df = df.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
        }
    )

    # Convert date
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")


    # Add season string
    df["season"] = season_str

    # Sort by date within season
    df = df.sort_values("date").reset_index(drop=True)

    # Approximate gameweek: 10 matches per round
    # For PL, 10 matches each gameweek → index // 10 + 1
    df["gameweek"] = (df.index // 10) + 1

    # Reorder columns to match our schema (match_id added later)
    df = df[
        [
            "season",
            "gameweek",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        ]
    ]

    return df


def build_all_matches():
    # 1. Load all historical seasons
    hist_dfs = []
    for season_str, filename in HIST_SEASONS:
        print(f"Loading season {season_str} from {filename}...")
        season_df = load_historical_season(season_str, filename)
        hist_dfs.append(season_df)

    hist_all = pd.concat(hist_dfs, ignore_index=True)
    print(f"Historical matches loaded: {len(hist_all)}")

    # 2. Load your current season (e.g. 2025-2026) from existing matches_master.csv
    if CURRENT_MATCHES_PATH.exists():
        current_df = pd.read_csv(CURRENT_MATCHES_PATH, parse_dates=["date"])
        # We only append if its season is not already in historical
        current_df = current_df[
            ~current_df["season"].isin([s for s, _ in HIST_SEASONS])
        ]
        current_df = current_df[
            [
                "season",
                "gameweek",
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
            ]
        ]
        print(f"Current season matches loaded: {len(current_df)}")
        all_matches = pd.concat([hist_all, current_df], ignore_index=True)
    else:
        print("No existing matches_master.csv found for current season. Using historical only.")
        all_matches = hist_all

    # 3. Sort by season + date
    all_matches = all_matches.sort_values(["season", "date"]).reset_index(drop=True)

    # 4. Assign new match_id
    all_matches["match_id"] = all_matches.index

    # 5. Reorder columns
    all_matches = all_matches[
        [
            "match_id",
            "season",
            "gameweek",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        ]
    ]

    # 6. Save back to matches_master.csv (overwrite)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_matches.to_csv(OUT_PATH, index=False)

    print(f"Saved combined matches to {OUT_PATH}")
    print(f"Total matches: {len(all_matches)}")
    print("Seasons included:", sorted(all_matches['season'].unique()))


if __name__ == "__main__":
    build_all_matches()
