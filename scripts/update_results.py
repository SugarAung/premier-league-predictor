#!/usr/bin/env python
"""
Fetch Premier League results + fixtures from football-data.org
and save them into data/live/ for the Streamlit app.

Outputs:
- data/live/pl_results_from_api.csv      (finished matches with actual_result)
- data/live/pl_fixtures_from_api.csv     (upcoming / not-finished fixtures)
"""

import requests
import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
# ⚠️ PUT YOUR REAL TOKEN HERE LOCALLY, BUT DO NOT COMMIT IT TO GITHUB
API_TOKEN = Path("my_api_key.txt").read_text().strip()
BASE_URL = "https://api.football-data.org/v4"

ROOT_DIR = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT_DIR / "data" / "live"


def fetch_premier_league_matches():
    """
    Fetch matches for the CURRENT Premier League season
    from football-data.org.
    """
    url = f"{BASE_URL}/competitions/PL/matches"
    headers = {"X-Auth-Token": API_TOKEN}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    return data["matches"]  # list of match dicts


def result_from_score(home_goals, away_goals):
    """Convert goals into H / D / A."""
    if home_goals > away_goals:
        return "H"
    elif home_goals < away_goals:
        return "A"
    else:
        return "D"


def build_results_dataframe(matches):
    """
    Finished matches → results dataframe:
    season, gameweek, date, home_team, away_team,
    full_time_home_goals, full_time_away_goals, actual_result
    """
    rows = []

    for m in matches:
        status = m.get("status")
        if status != "FINISHED":
            continue

        full_time = m["score"]["fullTime"]
        home_goals = full_time["home"]
        away_goals = full_time["away"]

        # Skip weird partial / missing scores
        if home_goals is None or away_goals is None:
            continue

        winner = m["score"].get("winner")
        if home_goals == 0 and away_goals == 0 and winner is None:
            # Probably not actually played yet
            continue

        utc_date = m.get("utcDate", "")[:10]
        season_start = m["season"]["startDate"][:4]
        season_end = m["season"]["endDate"][:4]
        season_str = f"{season_start}-{season_end}"

        row = {
            "season": season_str,
            "gameweek": m.get("matchday"),
            "date": utc_date,
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "full_time_home_goals": home_goals,
            "full_time_away_goals": away_goals,
            "actual_result": result_from_score(home_goals, away_goals),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_fixtures_dataframe(matches):
    """
    Upcoming / not-finished matches → fixtures dataframe:
    season, gameweek, date, home_team, away_team, status
    """
    rows = []

    for m in matches:
        status = m.get("status")
        # statuses like SCHEDULED, TIMED, POSTPONED, etc.
        if status == "FINISHED":
            continue

        utc_date = m.get("utcDate", "")[:10]
        season_start = m["season"]["startDate"][:4]
        season_end = m["season"]["endDate"][:4]
        season_str = f"{season_start}-{season_end}"

        row = {
            "season": season_str,
            "gameweek": m.get("matchday"),
            "date": utc_date,
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "status": status,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("Fetching current Premier League season from API...")
    matches = fetch_premier_league_matches()
    print(f"Total matches returned by API: {len(matches)}")

    # Ensure live folder exists
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Finished matches
    df_results = build_results_dataframe(matches)
    print(f"Finished matches found: {len(df_results)}")
    results_path = LIVE_DIR / "pl_results_from_api.csv"
    df_results.to_csv(results_path, index=False)
    print(f"Saved finished results to {results_path}")

    # Upcoming fixtures
    df_fixtures = build_fixtures_dataframe(matches)
    print(f"Upcoming / not-finished matches: {len(df_fixtures)}")
    fixtures_path = LIVE_DIR / "pl_fixtures_from_api.csv"
    df_fixtures.to_csv(fixtures_path, index=False)
    print(f"Saved fixtures to {fixtures_path}")


if __name__ == "__main__":
    main()
