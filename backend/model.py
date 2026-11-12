
"""
Power Generation Forecasting Engine
-----------------------------------
Loads operational data, trains regression models,
evaluates forecast performance and predicts generation.

Target: actual_mwh
Baseline: forecast_mwh
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "generation_forecast.csv"

TARGET = "actual_mwh"

BASE_FEATURES = [
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "outage_flag",
    "forecast_mwh"
]

REQUIRED_COLUMNS = [
    "date",
    *BASE_FEATURES,
    TARGET
]


# --------------------------------------------------
# MODEL CLASS
# --------------------------------------------------

class GenerationForecastingModel:

    def __init__(self, data_path=DATA_PATH):

        self.data_path = Path(data_path)

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.model_name = None

        self.results = {}
        self.feature_importance = {}

        self.trained = False

        self.feature_columns = [
            *BASE_FEATURES,
            "irradiance_temperature",
            "wind_irradiance"
        ]

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    def load_data(self):

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        df = pd.read_csv(self.data_path)

        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        df = df[REQUIRED_COLUMNS].copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        numeric_columns = BASE_FEATURES + [TARGET]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(subset=REQUIRED_COLUMNS)

        if df.empty:
            raise ValueError("Dataset contains no valid records.")

        # Validate operational values
        if (df[["irradiance_wm2", "wind_speed_ms"]] < 0).any().any():
            raise ValueError(
                "Irradiance and wind speed cannot be negative."
            )

        if not df["outage_flag"].isin([0, 1]).all():
            raise ValueError(
                "outage_flag must contain only 0 or 1."
            )

        if (df[TARGET] < 0).any():
            raise ValueError(
                "Actual generation cannot be negative."
            )

        self.data = df.reset_index(drop=True)

        return self.data

    # --------------------------------------------------
    # 2. FEATURE ENGINEERING
    # --------------------------------------------------

    def engineer_features(self, df):

        features = df[BASE_FEATURES].copy()

        # Interaction between solar irradiation and temperature
        features["irradiance_temperature"] = (
            features["irradiance_wm2"]
            * features["temperature_c"]
        ) / 1000

        # Interaction between irradiation and wind speed
        features["wind_irradiance"] = (
            features["irradiance_wm2"]
            * features["wind_speed_ms"]
        ) / 1000

        return features[self.feature_columns]

    # --------------------------------------------------
    # 3. DEFINE MODELS
    # --------------------------------------------------

    def get_models(self):

        return {

            "Linear Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ]),

            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        }

    # --------------------------------------------------
    # 4. CALCULATE METRICS
    # --------------------------------------------------

    @staticmethod
    def calculate_metrics(actual, predicted):

        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        return {
            "mae": round(
                float(mean_absolute_error(actual, predicted)), 4
            ),

            "rmse": round(
                float(np.sqrt(mean_squared_error(actual, predicted))), 4
            ),

            "mape": round(
                float(mean_absolute_percentage_error(
                    actual, predicted
                ) * 100), 2
            ),

            "r2": round(
                float(r2_score(actual, predicted)), 4
            )
        }

    # --------------------------------------------------
    # 5. TRAIN AND EVALUATE
    # --------------------------------------------------

    def train(self):

        if self.data is None:
            self.load_data()

        X = self.engineer_features(self.data)
        y = self.data[TARGET]

        # Keep observations from the same date together.
        # This avoids splitting same-date records across
        # training and testing sets.

        groups = self.data["date"].dt.strftime("%Y-%m-%d")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42
        )

        train_idx, test_idx = next(
            splitter.split(X, y, groups)
        )

        self.X_train = X.iloc[train_idx]
        self.X_test = X.iloc[test_idx]

        self.y_train = y.iloc[train_idx]
        self.y_test = y.iloc[test_idx]

        train_groups = groups.iloc[train_idx]

        models = self.get_models()

        # Group-aware cross-validation on training data
        n_groups = train_groups.nunique()
        cv = GroupKFold(n_splits=min(5, n_groups))

        cv_results = {}

        for name, candidate in models.items():

            scores = cross_val_score(
                candidate,
                self.X_train,
                self.y_train,
                groups=train_groups,
                cv=cv,
                scoring="neg_mean_absolute_error"
            )

            cv_results[name] = abs(float(scores.mean()))

        # Select model using training-only CV results
        self.model_name = min(
            cv_results,
            key=cv_results.get
        )

        self.model = models[self.model_name]

        self.model.fit(
            self.X_train,
            self.y_train
        )

        # Evaluate on untouched holdout data
        predictions = self.model.predict(self.X_test)

        baseline_predictions = self.X_test["forecast_mwh"]

        self.results = {
            "selected_model": self.model_name,

            "training_records": int(len(self.X_train)),

            "testing_records": int(len(self.X_test)),

            "cross_validation_mae": {
                name: round(score, 4)
                for name, score in cv_results.items()
            },

            "ml_model": self.calculate_metrics(
                self.y_test,
                predictions
            ),

            "existing_forecast": self.calculate_metrics(
                self.y_test,
                baseline_predictions
            )
        }

        self._calculate_feature_importance()

        self.trained = True

        return self.results

    # --------------------------------------------------
    # 6. FEATURE IMPORTANCE
    # --------------------------------------------------

    def _calculate_feature_importance(self):

        if self.model_name == "Linear Regression":

            coefficients = (
                self.model.named_steps["regressor"].coef_
            )

            importance = np.abs(coefficients)

        else:

            importance = self.model.feature_importances_

        total = importance.sum()

        if total == 0:
            importance = np.zeros(len(self.feature_columns))
        else:
            importance = importance / total * 100

        self.feature_importance = {
            name: round(float(value), 2)
            for name, value in zip(
                self.feature_columns,
                importance
            )
        }

        self.feature_importance = dict(
            sorted(
                self.feature_importance.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

    # --------------------------------------------------
    # 7. PREDICT GENERATION
    # --------------------------------------------------

    def predict_generation(self, input_data):

        if not self.trained:
            raise RuntimeError(
                "Model has not been trained. Call train() first."
            )

        missing = set(BASE_FEATURES) - set(input_data.keys())

        if missing:
            raise ValueError(
                f"Missing prediction inputs: {sorted(missing)}"
            )

        row = pd.DataFrame(
            [{column: input_data[column] for column in BASE_FEATURES}]
        )

        # Validate new operational inputs
        for column in BASE_FEATURES:
            row[column] = pd.to_numeric(
                row[column],
                errors="raise"
            )

        if not np.isfinite(row[BASE_FEATURES].to_numpy()).all():
            raise ValueError("Inputs must contain finite numbers.")

        if row[["irradiance_wm2", "wind_speed_ms"]].lt(0).any().any():
            raise ValueError("Irradiance and wind speed cannot be negative.")

        if row["temperature_c"].lt(-50).any() or row["temperature_c"].gt(80).any():
            raise ValueError("Temperature must be between -50 and 80 C.")

        if row["irradiance_wm2"].gt(1500).any():
            raise ValueError("Irradiance exceeds the supported input range.")

        if row["wind_speed_ms"].gt(100).any():
            raise ValueError("Wind speed exceeds the supported input range.")

        if not row["outage_flag"].isin([0, 1]).all():
            raise ValueError("outage_flag must be 0 or 1.")

        if row["forecast_mwh"].lt(0).any():
            raise ValueError("Forecast generation cannot be negative.")

        features = self.engineer_features(row)

        prediction = float(
            self.model.predict(features)[0]
        )

        prediction = max(0.0, prediction)

        forecast = float(row["forecast_mwh"].iloc[0])

        variance = forecast - prediction

        variance_percentage = (
            (variance / forecast) * 100
            if forecast != 0 else None
        )

        return {
            "predicted_generation_mwh": round(prediction, 3),

            "existing_forecast_mwh": round(forecast, 3),

            "forecast_minus_prediction_mwh": round(
                variance, 3
            ),

            "variance_percentage": (
                round(variance_percentage, 2)
                if variance_percentage is not None else None
            )
        }

    # --------------------------------------------------
    # 8. GET MODEL SUMMARY
    # --------------------------------------------------

    def get_summary(self):

        if not self.trained:
            raise RuntimeError(
                "Train the model before requesting summary."
            )

        return {
            "model_name": self.model_name,

            "dataset_records": int(len(self.data)),

            "metrics": self.results,

            "feature_importance": self.feature_importance,

            "limitations": [
                "Dataset contains only a small number of observations.",
                "Observations are grouped by date for holdout validation.",
                "The model estimates actual generation using an existing forecast.",
                "Results require validation on larger real-world datasets."
            ]
        }


# --------------------------------------------------
# LOCAL EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    engine = GenerationForecastingModel()

    print("\nLoading dataset...")

    dataset = engine.load_data()

    print("Dataset shape:", dataset.shape)

    print("\nTraining forecasting models...")

    results = engine.train()

    print("\nMODEL EVALUATION")

    for key, value in results.items():
        print(f"\n{key}: {value}")

    print("\nFEATURE IMPORTANCE")

    for feature, importance in engine.feature_importance.items():
        print(f"{feature}: {importance}%")

    print("\nMODEL TRAINING COMPLETED!")
