from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

def run_analysis():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "fema_merged.csv")

    print(df.shape)
    print(df.head())
    print(df.columns)

    by_state = (
        df.groupby("state_pa")["federalShareObligated"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    by_incident_type = (
        df.groupby("incidentType")["federalShareObligated"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    by_state.to_csv(REPORTS_DIR / "fema_funding_by_state.csv", index=False)
    by_incident_type.to_csv(REPORTS_DIR / "fema_funding_by_incident_type.csv", index=False)

    print("Analysis complete")

if __name__ == "__main__":
    run_analysis()