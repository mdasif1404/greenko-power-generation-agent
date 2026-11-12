
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path

from backend.model import GenerationForecastingModel, BASE_FEATURES


app = FastAPI(title='Power Generation Forecasting & Variance Explanation Agent')

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
FRONTEND = BASE / 'frontend' / 'index.html'

engine = None


# Input schema

class Query(BaseModel):
    text: str = ''


class AnalysisRequest(BaseModel):
    irradiance_wm2: float = Field(ge=0, le=1500)
    wind_speed_ms: float = Field(ge=0, le=100)
    temperature_c: float = Field(ge=-50, le=80)
    outage_flag: int = Field(ge=0, le=1)
    forecast_mwh: float = Field(ge=0)
    actual_mwh: float | None = Field(default=None, ge=0)


# Initialize forecasting model

@app.on_event('startup')
def startup():

    global engine

    try:
        engine = GenerationForecastingModel(DATA / 'generation_forecast.csv')
        engine.load_data()
        engine.train()

        print('Forecasting model initialized successfully.')

    except Exception as e:
        print(f'Model initialization failed: {e}')


def get_engine():

    if engine is None or not engine.trained:
        raise HTTPException(
            status_code=503,
            detail='Forecasting model is not available.'
        )

    return engine


# Serve frontend

@app.get('/', include_in_schema=False)
def home():

    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail='Frontend not found.')

    return FileResponse(FRONTEND)


# Health check

@app.get('/api/health')
def health():

    return {
        'status': 'healthy' if engine and engine.trained else 'unavailable',
        'model_loaded': engine is not None and engine.trained
    }


# Demo endpoint

@app.get('/api/demo')
def demo():

    model = get_engine()

    return {
        'case_study': 'Power Generation Forecasting & Variance Explanation Agent',
        'status': 'running',
        'model': model.model_name,
        'expected_outcome': 'Forecast generation, analyze variance and explain possible generation loss drivers.',
        'features': BASE_FEATURES
    }


# Model evaluation

@app.get('/api/metrics')
def metrics():

    model = get_engine()

    return model.get_summary()


# Historical generation data

@app.get('/api/history')
def history():

    model = get_engine()

    df = model.data.copy()

    records = []

    for _, row in df.iterrows():

        records.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'forecast_mwh': round(float(row['forecast_mwh']), 2),
            'actual_mwh': round(float(row['actual_mwh']), 2),
            'variance_mwh': round(
                float(row['actual_mwh'] - row['forecast_mwh']), 2
            ),
            'irradiance_wm2': float(row['irradiance_wm2']),
            'wind_speed_ms': float(row['wind_speed_ms']),
            'temperature_c': float(row['temperature_c']),
            'outage_flag': int(row['outage_flag'])
        })

    return {
        'total_records': len(records),
        'records': records
    }


# Retrieve similar historical records

def retrieve_similar_records(input_data, top_k=5):

    model = get_engine()

    df = model.data.copy()

    columns = BASE_FEATURES

    historical = df[columns].astype(float)

    query = np.array([
        input_data[column] for column in columns
    ], dtype=float)

    means = historical.mean().values
    stds = historical.std().replace(0, 1).values

    historical_scaled = (historical.values - means) / stds
    query_scaled = (query - means) / stds

    distances = np.sqrt(
        np.sum((historical_scaled - query_scaled) ** 2, axis=1)
    )

    nearest = np.argsort(distances)[:top_k]

    results = []

    for index in nearest:

        row = df.iloc[index]

        results.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'similarity_distance': round(float(distances[index]), 4),
            'forecast_mwh': float(row['forecast_mwh']),
            'actual_mwh': float(row['actual_mwh']),
            'variance_mwh': round(
                float(row['actual_mwh'] - row['forecast_mwh']), 2
            ),
            'irradiance_wm2': float(row['irradiance_wm2']),
            'wind_speed_ms': float(row['wind_speed_ms']),
            'temperature_c': float(row['temperature_c']),
            'outage_flag': int(row['outage_flag'])
        })

    return results


# Variance explanation agent

def generate_explanation(input_data, prediction, similar_records):

    model = get_engine()
    df = model.data

    factors = []
    recommendations = []

    irradiance = input_data['irradiance_wm2']
    wind = input_data['wind_speed_ms']
    temperature = input_data['temperature_c']
    outage = input_data['outage_flag']

    forecast = input_data['forecast_mwh']
    predicted = prediction['predicted_generation_mwh']

    # Compare operational conditions with historical medians

    if irradiance < df['irradiance_wm2'].median():

        factors.append({
            'factor': 'Low solar irradiation',
            'observation': 'Irradiation is below the historical median.',
            'impact': 'Potential reduction in solar generation'
        })

        recommendations.append(
            'Investigate cloud cover and solar panel exposure.'
        )

    if temperature > df['temperature_c'].median():

        factors.append({
            'factor': 'High temperature',
            'observation': 'Temperature is above the historical median.',
            'impact': 'May reduce photovoltaic panel efficiency'
        })

        recommendations.append(
            'Inspect module temperature and cooling conditions.'
        )

    if wind < df['wind_speed_ms'].median():

        factors.append({
            'factor': 'Low wind speed',
            'observation': 'Wind speed is below the historical median.',
            'impact': 'Potential reduction in wind generation'
        })

        recommendations.append(
            'Investigate wind turbine operating conditions.'
        )

    if outage == 1:

        factors.append({
            'factor': 'Equipment outage',
            'observation': 'An outage is recorded in the input.',
            'impact': 'Potential generation loss'
        })

        recommendations.append(
            'Inspect outage logs and equipment availability.'
        )

    # Forecast variance

    predicted_variance = predicted - forecast

    if predicted_variance < 0:

        factors.append({
            'factor': 'Forecast overestimation',
            'observation': 'Model prediction is below existing forecast.',
            'impact': 'Potential generation shortfall'
        })

    elif predicted_variance > 0:

        factors.append({
            'factor': 'Forecast underestimation',
            'observation': 'Model prediction is above existing forecast.',
            'impact': 'Potential generation above forecast'
        })

    # Historical evidence

    if similar_records:

        historical_variance = np.mean([
            record['variance_mwh']
            for record in similar_records
        ])

        factors.append({
            'factor': 'Historical comparison',
            'observation': (
                f'{len(similar_records)} similar records were retrieved. '
                f'Their average actual-minus-forecast variance is '
                f'{historical_variance:.2f} MWh.'
            ),
            'impact': 'Provides historical context for the current conditions'
        })

    if not factors:

        factors.append({
            'factor': 'No significant variance driver detected',
            'observation': 'No configured operational indicator was triggered.',
            'impact': 'Further investigation may be required'
        })

    if not recommendations:

        recommendations.append(
            'Continue monitoring generation and operational conditions.'
        )

    # Actual generation comparison, when provided

    actual = input_data.get('actual_mwh')

    actual_variance = None

    if actual is not None:
        actual_variance = actual - forecast

    return {
        'result': (
            'Generation shortfall' if actual_variance is not None
            and actual_variance < 0 else
            'Generation above forecast' if actual_variance is not None
            and actual_variance > 0 else
            'Forecast matched actual generation' if actual_variance == 0
            else 'Forecast overestimation' if predicted_variance < 0
            else 'Forecast underestimation' if predicted_variance > 0
            else 'Forecast matched model prediction'
        ),

        'predicted_variance_mwh': round(predicted_variance, 2),

        'actual_variance_mwh': (
            round(actual_variance, 2)
            if actual_variance is not None else None
        ),

        'variance_drivers': factors,

        'recommendations': recommendations,

        'feature_importance': model.feature_importance,

        'explanation_note': (
            'These are potential variance drivers based on historical '
            'comparisons and operational indicators, not causal proof.'
        )
    }


# Main forecasting and variance analysis endpoint

@app.post('/api/analyze')
def analyze(request: AnalysisRequest):

    model = get_engine()

    input_data = request.dict()

    try:

        # Generate prediction

        prediction = model.predict_generation(input_data)

        # Retrieve similar historical records

        similar_records = retrieve_similar_records(input_data)

        # Generate explanation

        explanation = generate_explanation(
            input_data,
            prediction,
            similar_records
        )

        return {
            'status': 'success',
            'input': input_data,
            'forecast': prediction,
            'explanation': explanation,
            'historical_evidence': similar_records
        }

    except (ValueError, RuntimeError) as e:

        raise HTTPException(status_code=400, detail=str(e))


# Original run endpoint - enhanced

@app.post('/api/run')
def run_case(query: Query):

    model = get_engine()

    return {
        'input': query.text,
        'result': 'Forecasting and variance explanation agent is running.',
        'model': model.model_name,
        'evaluation': model.results,
        'feature_importance': model.feature_importance,
        'limitations': model.get_summary()['limitations']
    }
