import requests
import pandas as pd

API_TOKEN = "3517c5add97f4ce299acef91844b7585"  # <-- your token
BASE_URL = "https://api.football-data.org/v4"


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

        if home_goals is None or away_goals is None:
            continue

        winner = m["score"].get("winner")
        if home_goals == 0 and away_goals == 0 and winner is None:
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

    # Finished matches
    df_results = build_results_dataframe(matches)
    print(f"Finished matches found: {len(df_results)}")
    df_results.to_csv("pl_results_from_api.csv", index=False)
    print("Saved finished results to pl_results_from_api.csv")

    # Upcoming fixtures
    df_fixtures = build_fixtures_dataframe(matches)
    print(f"Upcoming / not-finished matches: {len(df_fixtures)}")
    df_fixtures.to_csv("pl_fixtures_from_api.csv", index=False)
    print("Saved fixtures to pl_fixtures_from_api.csv")


if __name__ == "__main__":
    main()
