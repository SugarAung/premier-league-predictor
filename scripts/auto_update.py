import subprocess
import os

def run(cmd):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("\n====================================")
    print("    AUTOMATED UPDATE (GitHub Actions)")
    print("====================================\n")

    # Write API key to the file your script expects
    # (update_results.py reads from my_api_key.txt)
    if "API_KEY" in os.environ:
        with open("my_api_key.txt", "w") as f:
            f.write(os.environ["API_KEY"])

    # 1. Fetch updated results & fixtures
    run("python scripts/update_results.py")

    # 2. Run prediction + archive pipeline
    run("python scripts/pipeline.py")

    print("\n✔ Auto update script completed!\n")

if __name__ == "__main__":
    main()
