# Demo Notes
## Power Generation Forecasting & Variance Explanation Agent

### 1. Live Application

**Dashboard:**
https://greenko-power-generation-agent-production.up.railway.app

**API Documentation:**
https://greenko-power-generation-agent-production.up.railway.app/docs

**GitHub Repository:**
https://github.com/mdasif1404/greenko-power-generation-agent

### 2. Running the Application Locally

Requirements:
- Python 3.10 or compatible version.
- Git.
- VS Code or another Python editor.

Clone the repository:

```bash
git clone https://github.com/mdasif1404/greenko-power-generation-agent.git
cd greenko-power-generation-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn backend.app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

### 3. Dashboard Demonstration

The dashboard provides:

1. Selected forecasting model.
2. Number of dataset observations.
3. Model MAE and R².
4. Comparison between ML model and existing forecast.
5. Input form for operating conditions.
6. Generation prediction.
7. Variance explanation.
8. Operational recommendations.
9. Similar historical observations.
10. Feature importance and limitations.

### 4. Example Input

The following input can be used to demonstrate the analysis endpoint:

```json
{
  "irradiance_wm2": 650,
  "wind_speed_ms": 5.2,
  "temperature_c": 32,
  "outage_flag": 0,
  "forecast_mwh": 48
}
```

Actual generation can optionally be supplied for retrospective variance analysis.

### 5. Example Output

An example model response from the local test:

- Predicted generation: 48.463 MWh
- Existing forecast: 48 MWh
- Predicted variance: +0.463 MWh
- Variance percentage: approximately +0.96%

The explanation component identifies potential operating-condition drivers and retrieves similar historical records.

The exact output can vary depending on the input conditions.

### 6. Example Test Scenarios

**Scenario 1: Normal operating conditions**

- Irradiance: 750 W/m²
- Wind speed: 7 m/s
- Temperature: 28°C
- Outage: No
- Forecast: 55 MWh

**Scenario 2: Low wind conditions**

- Irradiance: 500 W/m²
- Wind speed: 2 m/s
- Temperature: 35°C
- Outage: No
- Forecast: 45 MWh

**Scenario 3: Equipment outage**

- Irradiance: 650 W/m²
- Wind speed: 5 m/s
- Temperature: 32°C
- Outage: Yes
- Forecast: 48 MWh

These are demonstration inputs, not verified operational scenarios or guaranteed prediction outcomes.

### 7. API Testing

The following endpoints can be tested using the browser or Swagger UI:

| Endpoint | Purpose |
|---|---|
| `/api/health` | Check backend and model status |
| `/api/metrics` | Inspect model evaluation |
| `/api/history` | Inspect historical observations |
| `/api/analyze` | Submit generation analysis |
| `/docs` | Interactive API documentation |

Example health response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 8. Evaluation Summary

| Metric | ML Model | Existing Forecast |
|---|---:|---:|
| MAE | 4.463 | 5.380 |
| RMSE | 5.2394 | 5.5928 |
| MAPE | 8.97% | 10.59% |
| R² | 0.8985 | 0.8843 |

The selected Linear Regression model achieved approximately 17.04% lower MAE than the existing forecast on the held-out test set.

### 9. Deployment Verification

The application was deployed to Railway.

Deployment verification included:

- Successful dependency installation.
- Successful application startup.
- Successful forecasting model initialization.
- Successful health endpoint response.
- Successful model metrics endpoint response.
- Successful live dashboard loading.
- Successful live generation analysis demonstration.

### 10. Important Notes

- The application uses the bundled assessment dataset.
- Predictions are estimates, not guaranteed generation values.
- Variance explanations identify potential contributors, not confirmed causes.
- The actual generation input is optional.
- The project currently uses deterministic rule-based explanations rather than an integrated external LLM.
- The small dataset limits generalization and production readiness.