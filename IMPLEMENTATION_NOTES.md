# Implementation Notes
## Power Generation Forecasting & Variance Explanation Agent

### 1. Project Overview

This project implements a machine-learning-based renewable power generation forecasting and variance explanation application.

The objective is to estimate actual power generation using historical generation data and operating conditions, compare the prediction against the existing forecast, and identify potential factors contributing to generation variance.

The application provides:
- Machine learning-based generation prediction.
- Comparison between predicted and existing forecast generation.
- Actual-versus-forecast variance analysis when actual generation is supplied.
- Retrieval of similar historical observations.
- Rule-based variance explanations.
- Operational recommendations.
- An interactive web dashboard.

### 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend framework | FastAPI |
| Machine learning | Scikit-learn |
| Data processing | Pandas, NumPy |
| Regression algorithms | Linear Regression, Random Forest |
| Data validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Railway |
| Version control | Git and GitHub |

### 3. Dataset Description

The application uses the bundled dummy dataset provided with the assessment.

Dataset characteristics:
- Total observations: 60
- Observation period: August 2026
- Records per day: 2
- Total columns: 7

Features:

| Feature | Description |
|---|---|
| date | Observation date |
| irradiance_wm2 | Solar irradiance |
| wind_speed_ms | Wind speed |
| temperature_c | Ambient temperature |
| outage_flag | Equipment outage indicator |
| forecast_mwh | Existing generation forecast |
| actual_mwh | Actual generation |

The dataset is small and intended for demonstration and baseline model evaluation.

### 4. Data Preprocessing

Data preprocessing and validation are performed in the Python model pipeline.

The implementation:
1. Loads the CSV dataset.
2. Validates the required columns.
3. Converts date values into datetime format.
4. Converts numerical features into numeric types.
5. Removes incomplete records.
6. Checks for invalid negative irradiance, wind speed and actual generation values.
7. Validates the binary outage indicator.

### 5. Feature Engineering

Two interaction features are created:

- `irradiance_temperature`: captures the interaction between solar irradiance and temperature.
- `wind_irradiance`: captures the interaction between wind speed and solar irradiance.

The original operating variables and existing forecast are retained as model inputs.

### 6. Machine Learning Approach

Two regression algorithms are evaluated:

**Linear Regression**
- Provides an interpretable baseline.
- Uses feature scaling in a machine learning pipeline.
- Models relationships between operating conditions and actual generation.

**Random Forest Regressor**
- Captures nonlinear relationships.
- Uses an ensemble of decision trees.
- Provides feature importance values.

The model with the lower cross-validation MAE is selected.

Linear Regression was selected for the current evaluation.

### 7. Training and Validation

The dataset is divided using date-grouped validation to reduce the risk of observations from the same date appearing in both training and testing sets.

Evaluation configuration:
- Training records: 48
- Testing records: 12
- Holdout split: 80:20
- Cross-validation: GroupKFold
- Grouping variable: date

The existing forecast is also evaluated on the same held-out test records to provide a baseline comparison.

### 8. Model Evaluation Results

| Metric | Linear Regression | Existing Forecast |
|---|---:|---:|
| MAE | 4.463 | 5.380 |
| RMSE | 5.2394 | 5.5928 |
| MAPE | 8.97% | 10.59% |
| R² | 0.8985 | 0.8843 |

The selected machine learning model achieved approximately 17.04% lower MAE than the existing forecast on the held-out test set.

These results should be interpreted cautiously because the test set contains only 12 observations.

### 9. Forecasting and Variance Analysis

The model estimates actual generation using operating conditions and the existing forecast as input features.

Two comparisons are provided:

**Predicted variance**

Predicted generation minus existing forecast.

A positive value indicates predicted generation above the existing forecast. A negative value indicates predicted generation below the existing forecast.

**Actual variance**

Actual generation minus existing forecast.

This comparison is available when actual generation is supplied.

Separating these two measurements helps distinguish prospective prediction from retrospective performance analysis.

### 10. Historical Retrieval

The application retrieves five similar historical observations.

The retrieval workflow:
1. Uses operating conditions and existing forecast as input features.
2. Standardizes feature differences using historical data statistics.
3. Calculates Euclidean distance between the input and historical observations.
4. Selects the five nearest records.
5. Displays their operating conditions and actual-versus-forecast variance.

The retrieved records provide historical context for the explanation.

This is a lightweight similarity-based retrieval mechanism rather than a vector-database-based RAG implementation.

### 11. Variance Explanation Agent

The explanation component uses rule-based operational indicators and historical comparisons.

It examines:
- Solar irradiance relative to historical conditions.
- Wind speed relative to historical conditions.
- Temperature relative to historical conditions.
- Equipment outage status.
- Predicted generation relative to the existing forecast.
- Variance patterns in similar historical observations.

Based on these indicators, the application generates:
- Potential variance drivers.
- Observations.
- Potential operational impacts.
- Suggested recommendations.

The system uses deterministic Python rules. An external LLM is not currently integrated.

The explanations are intended as decision-support insights, not proof of causation.

### 12. Backend API

The application uses FastAPI.

Main endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the frontend dashboard |
| `/api/health` | GET | Checks application and model status |
| `/api/demo` | GET | Provides demonstration data |
| `/api/metrics` | GET | Returns model evaluation results |
| `/api/history` | GET | Returns historical observations |
| `/api/analyze` | POST | Performs prediction and variance analysis |
| `/api/run` | POST | Runs the demonstration workflow |

Interactive API documentation is available through `/docs`.

### 13. Frontend

The frontend is implemented using HTML, CSS and JavaScript.

Dashboard functionality includes:
- Model performance overview.
- Generation analysis input form.
- Predicted generation and forecast variance.
- Variance explanations.
- Operational recommendations.
- Similar historical observations.
- Feature importance.
- Dataset limitations.

The frontend communicates with the backend using HTTP requests.

### 14. Assumptions and Limitations

1. The bundled dataset contains only 60 observations.
2. The dataset is synthetic/dummy assessment data and may not represent actual renewable asset behaviour.
3. The model uses an existing forecast as an input feature.
4. The dataset does not identify individual plants or assets.
5. The model is evaluated using only 12 held-out records.
6. Similarity retrieval provides historical context but does not establish causation.
7. Rule-based explanations are not equivalent to an LLM-generated operational report.
8. Feature importance should not be interpreted as causal influence.
9. The current model does not explicitly model seasonality or long-term time-series trends.
10. Results require validation on larger real-world generation datasets.

### 15. Production Readiness and Future Improvements

Potential improvements include:

- Collecting larger datasets from actual renewable assets.
- Developing separate models for solar and wind generation.
- Adding asset-level and location-level features.
- Incorporating weather forecasts and historical weather data.
- Evaluating time-series forecasting algorithms.
- Adding prediction intervals and uncertainty estimates.
- Improving retrieval with richer historical context.
- Integrating an optional LLM for natural-language summaries, with a deterministic fallback.
- Monitoring model drift and prediction accuracy.
- Loading pre-trained models instead of retraining on every application startup.
- Implementing authentication, logging and monitoring.

### 16. Deployment

The application is deployed using Railway.

**Live application:**
https://greenko-power-generation-agent-production.up.railway.app

**GitHub repository:**
https://github.com/mdasif1404/greenko-power-generation-agent

**API documentation:**
https://greenko-power-generation-agent-production.up.railway.app/docs

### 17. Conclusion

The project demonstrates an end-to-end machine learning workflow for renewable power generation forecasting and variance explanation.

It combines regression modelling, feature engineering, date-grouped validation, historical similarity retrieval, rule-based explanations and an interactive web dashboard.

The implementation provides a reproducible baseline that can be extended using larger real-world energy datasets and more advanced forecasting and explanation techniques.