import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# ---------------------------------------------
# Config – change these if needed
# ---------------------------------------------
PYTHON = sys.executable  # use current Python env

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Script paths
PREDICT_SCRIPT = SCRIPTS_DIR / "predict_hybrid.py"            # generates data/live/pl_predictions.csv
ARCHIVE_SCRIPT = SCRIPTS_DIR / "archive_live_predictions.py"  # appends to data/live/live_predictions_history.csv
BACKTEST_SCRIPT = SCRIPTS_DIR / "backtest_hybrid.py"          # backtest script (optional)

# Files
BACKTEST_OUTPUT = ROOT_DIR / "data" / "processed" / "backtest_hybrid.csv"
BACKTEST_MIN_SEASON = "2025-2026"  # current season for backtest

# Toggles
RUN_BACKTEST = False  # set True if you want to refresh backtest each run


def run_step(name, cmd):
    """Run a subprocess step with nice logging."""
    print("\n" + "=" * 60)
    print(f"[PIPELINE] Step: {name}")
    print(f"[PIPELINE] Command: {' '.join(map(str, cmd))}")
    print("=" * 60)
    subprocess.check_call(list(map(str, cmd)))
    print(f"[PIPELINE] Step '{name}' finished OK")


def main():
    print("\n" + "#" * 60)
    print("#  PREMIER LEAGUE MODEL PIPELINE")
    print("#  Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 60 + "\n")

    # 1) Generate model predictions for ALL upcoming fixtures
    if PREDICT_SCRIPT.exists():
        run_step(
            "Generate model predictions (all fixtures)",
            [
                PYTHON,
                PREDICT_SCRIPT,
                "--output",
                ROOT_DIR / "data" / "live" / "pl_predictions.csv",
            ],
        )
    else:
        print(f"[PIPELINE] ERROR: {PREDICT_SCRIPT} not found. Aborting.")
        return

    # 2) Archive newly finished gameweeks into data/live/live_predictions_history.csv
    if ARCHIVE_SCRIPT.exists():
        run_step(
            "Archive predictions for finished gameweeks",
            [PYTHON, ARCHIVE_SCRIPT],
        )
    else:
        print(f"[PIPELINE] WARNING: {ARCHIVE_SCRIPT} not found, skipping archive step.")

    # 3) (Optional) Refresh backtest for this season
    if RUN_BACKTEST:
        if BACKTEST_SCRIPT.exists():
            run_step(
                "Run backtest for this season",
                [
                    PYTHON,
                    BACKTEST_SCRIPT,
                    "--min_season",
                    BACKTEST_MIN_SEASON,
                    "--output",
                    BACKTEST_OUTPUT,
                ],
            )
        else:
            print(f"[PIPELINE] Skipping backtest: {BACKTEST_SCRIPT} not found.")

    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETE")
    print("#  Now you can run: streamlit run app/app.py")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
