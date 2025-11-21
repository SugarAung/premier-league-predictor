import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def build_matches_master():
    # 1. Load your existing results CSV
    # If your file is in the root, either move it to data/raw or change the path
    results_path = Path("pl_results_from_api.csv")  # change if needed
    df = pd.read_csv(results_path)

    # 👇 Adjust these column names to what you actually have
    # Example mappings – you will tweak them
    col_map = {
    "full_time_home_goals": "home_goals",
    "full_time_away_goals": "away_goals",
    "season": "season",
    "gameweek": "gameweek",
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
}


    # Rename columns that exist
    df = df.rename(columns=col_map)

    # Make sure the key columns exist – if any of these fails,
    # check the printed column names and adjust col_map.
    required_cols = [
        "date",
        "season",
        "gameweek",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    print("Columns in CSV:", df.columns.tolist())
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' after renaming.")

    # Convert types
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(str)
    df["gameweek"] = df["gameweek"].astype(int)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Add result column: H / D / A
    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        else:
            return "D"

    df["result"] = df.apply(get_result, axis=1)

    # Sort by date (important for later rolling stats)
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    # Add a match_id
    df["match_id"] = df.index

    # Reorder columns nicely
    df = df[
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

    # Save processed version
    out_path = PROCESSED_DIR / "matches_master.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} matches to {out_path}")

if __name__ == "__main__":
    build_matches_master()
