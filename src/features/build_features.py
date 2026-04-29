from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"


def load_data() -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / "fema_merged.parquet"
    csv_path = PROCESSED_DIR / "fema_merged.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    return pd.read_csv(csv_path)


def aggregate_to_disaster(df: pd.DataFrame) -> pd.DataFrame:
    print("Available columns:")
    print(df.columns.tolist())

    if "state_declaration" in df.columns:
        state_col = "state_declaration"
    elif "state_pa" in df.columns:
        state_col = "state_pa"
    else:
        state_col = "state"

    grouped = df.groupby("disasterNumber").agg(
        total_federal_obligated=("federalShareObligated", "sum"),
        num_projects=("federalShareObligated", "count"),
        state=(state_col, "first"),
        incidentType=("incidentType", "first"),
        fyDeclared=("fyDeclared", "first"),
        declarationDate=("declarationDate", "first"),
        incidentBeginDate=("incidentBeginDate", "first"),
        incidentEndDate=("incidentEndDate", "first"),
    ).reset_index()

    return grouped


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["declarationDate"] = pd.to_datetime(df["declarationDate"], errors="coerce")
    df["incidentBeginDate"] = pd.to_datetime(df["incidentBeginDate"], errors="coerce")
    df["incidentEndDate"] = pd.to_datetime(df["incidentEndDate"], errors="coerce")

    df["incident_duration_days"] = (
        df["incidentEndDate"] - df["incidentBeginDate"]
    ).dt.days

    df["incident_duration_days"] = (
        df["incident_duration_days"]
        .fillna(0)
        .clip(lower=0, upper=365)
    )

    df["declaration_year"] = df["declarationDate"].dt.year
    df["declaration_month"] = df["declarationDate"].dt.month
    df["declaration_quarter"] = df["declarationDate"].dt.quarter

    df["incidentType"] = df["incidentType"].astype("string").str.strip()

    df["is_hurricane"] = (df["incidentType"] == "Hurricane").astype(int)
    df["is_flood"] = (df["incidentType"] == "Flood").astype(int)
    df["is_fire"] = (df["incidentType"] == "Fire").astype(int)
    df["is_severe_storm"] = (df["incidentType"] == "Severe Storm").astype(int)

    df["cost_per_project"] = (
        df["total_federal_obligated"] / df["num_projects"].replace(0, pd.NA)
    )

    return df


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = aggregate_to_disaster(df)
    df = create_features(df)

    output_path = FEATURES_DIR / "fema_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")
    print(df.head())
    print(df.info())


if __name__ == "__main__":
    main()