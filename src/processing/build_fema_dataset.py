from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_declarations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_cols = ["declarationDate", "incidentBeginDate", "incidentEndDate"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "disasterNumber" in df.columns:
        df["disasterNumber"] = df["disasterNumber"].astype("Int64")

    df = df.drop_duplicates()

    return df


def clean_public_assistance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "dateObligated" in df.columns:
        df["dateObligated"] = pd.to_datetime(df["dateObligated"], errors="coerce")

    if "federalShareObligated" in df.columns:
        df["federalShareObligated"] = pd.to_numeric(
            df["federalShareObligated"], errors="coerce"
        )

    if "disasterNumber" in df.columns:
        df["disasterNumber"] = df["disasterNumber"].astype("Int64")

    df = df.drop_duplicates()

    return df


def run_build_fema_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    declarations = pd.read_csv(RAW_DIR / "declarations.csv")
    public_assistance = pd.read_csv(RAW_DIR / "public_assistance.csv")

    declarations_clean = clean_declarations(declarations)
    public_assistance_clean = clean_public_assistance(public_assistance)

    declarations_clean["disasterNumber"] = declarations_clean["disasterNumber"].astype("Int64")
    public_assistance_clean["disasterNumber"] = public_assistance_clean["disasterNumber"].astype("Int64")

    merged = public_assistance_clean.merge(
        declarations_clean,
        on="disasterNumber",
        how="left",
        suffixes=("_pa", "_declaration"),
    )

    merged.to_csv(PROCESSED_DIR / "fema_merged.csv", index=False)

    print("Merged dataset saved")


if __name__ == "__main__":
    run_build_fema_dataset()