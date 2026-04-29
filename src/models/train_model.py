from pathlib import Path
import os

import mlflow
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# load env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "features" / "fema_features.csv"

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "fema_cost_prediction")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)
    
def train():
    df = load_data()

    target = "total_federal_obligated"

    features = [
        "state",
        "incidentType",
        "fyDeclared",
        "incident_duration_days",
    ]

    # clean target
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # remove invalid rows
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    df = df.dropna(subset=features)

    X = df[features]
    y = np.log1p(df[target])

    categorical = ["state", "incidentType"]
    numeric = ["fyDeclared", "incident_duration_days"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        preds_exp = np.expm1(preds)
        y_test_exp = np.expm1(y_test)

        mae = mean_absolute_error(y_test_exp, preds_exp)
        rmse = np.sqrt(mean_squared_error(y_test_exp, preds_exp))
        r2 = r2_score(y_test_exp, preds_exp)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)

if __name__ == "__main__":
    train()