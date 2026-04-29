from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


def plot_top_states():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPORTS_DIR / "fema_funding_by_state.csv")

    top_states = df.head(10)

    plt.figure(figsize=(10, 6))
    plt.bar(top_states["state_pa"], top_states["federalShareObligated"])
    plt.title("Top 10 States by FEMA Public Assistance Funding")
    plt.xlabel("State")
    plt.ylabel("Federal Share Obligated")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "top_states_fema_funding.png")
    plt.close()

    print("Chart saved")


if __name__ == "__main__":
    plot_top_states()