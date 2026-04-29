# =============================================================================
# FIXED VERSION FOR RENDER DEPLOYMENT (SHAP DISABLED)
# =============================================================================

import pickle
import numpy as np
import os
# import shap   ❌ REMOVED (causing crash)
import pandas as pd
from django.conf import settings
from datetime import datetime

BASE_DIR  = settings.BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "super_admin", "ml")

with open(os.path.join(MODEL_DIR, "kerala_flood_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_config.pkl"), "rb") as f:
    config = pickle.load(f)

print(f"[ML MODEL] Loaded: {type(model).__name__}")

label_encoder    = config["label_encoder"]
target_encoder   = config["target_encoder"]
district_mapping = config["district_mapping"]
feature_columns  = config["feature_columns"]

# ❌ SHAP DISABLED
# _shap_explainer = shap.TreeExplainer(model)


DEMO_DISTRICT_OVERRIDES = {
    "Pathanamthitta": {"rain_1h": 120, "humidity": 92},
    "Wayanad": {"rain_1h": 90, "humidity": 88},
    "Idukki": {"rain_1h": 75, "humidity": 85}
}


def predict_flood_risk(district, weather, demo_mode=True, debug=False):

    print("\n" + "=" * 70)
    print("🤖 FLOOD RISK PREDICTION STARTED (NO SHAP)")
    print("=" * 70)

    if not weather:
        return {
            "risk": "Unknown",
            "confidence": {"Low": 0, "Medium": 0, "High": 0},
            "features": {},
            "explanation": []
        }

    weather = {
        "temperature": weather.get("temperature", 28),
        "humidity": weather.get("humidity", 70),
        "pressure": weather.get("pressure", 1010),
        "wind_speed": weather.get("wind_speed", 2),
        "rain_1h": weather.get("rain_1h", 0),
    }

    if demo_mode and district in DEMO_DISTRICT_OVERRIDES:
        override = DEMO_DISTRICT_OVERRIDES[district]
        for key, forced_value in override.items():
            weather[key] = max(weather.get(key, 0), forced_value)

    static_data = {
        "Elevation_m": 20,
        "River_Proximity": 0.5,
        "Drainage_Capacity": 0.5,
        "Population_Density": 1500
    }

    month = datetime.now().month
    rainfall = weather["rain_1h"]

    rainfall_7day = rainfall * 25
    rainfall_30day = rainfall * 80
    soil_moisture = min(95, rainfall_7day / 8)

    features = {
        "District_Encoded": district_mapping.get(district, 0),
        "Month": month,
        "Daily_Rainfall_mm": rainfall,
        "Rainfall_7day_mm": rainfall_7day,
        "Rainfall_30day_mm": rainfall_30day,
        "Wind_Speed_kmh": weather["wind_speed"] * 3.6,
        "Temperature_C": weather["temperature"],
        "Humidity_pct": weather["humidity"],
        "Pressure_hPa": weather["pressure"],
        **static_data,
        "Soil_Moisture_pct": soil_moisture,
        "Is_Monsoon": int(month in [6, 7, 8, 9]),
        "Is_Heavy_Rain": int(rainfall > 100),
        "Low_Elevation": 1,
        "Poor_Drainage": 0,
        "High_Soil_Moisture": int(soil_moisture > 70),
    }

    X_row = [features.get(col, 0) for col in feature_columns]
    X = pd.DataFrame([X_row], columns=feature_columns)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    risk_label = target_encoder.inverse_transform([pred])[0]

    class_labels = [target_encoder.inverse_transform([c])[0] for c in model.classes_]
    confidence = {label: round(prob * 100, 2)
                  for label, prob in zip(class_labels, probs)}

    confidence.setdefault("Low", 0.0)
    confidence.setdefault("Medium", 0.0)
    confidence.setdefault("High", 0.0)

    # ❌ SHAP REMOVED → returning empty explanation
    return {
        "risk": risk_label,
        "confidence": confidence,
        "features": features,
        "final_weather": weather,
        "explanation": [],   # empty instead of SHAP
        "base_value": 0,
        "shap_max": 1,
        "method": "Random Forest (SHAP Disabled for Deployment)",
    }
