"""
app.py — API Flask para predicción de precios de diamantes.

Local:
    python app.py

Producción (gunicorn):
    gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ── Cargar modelo al iniciar ─────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/housing_price_model.pkl")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
metrics  = artifact["metrics"]

# ── Validación ───────────────────────────────────────────────────────────────


REQUIRED_NUMERIC = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]


def validate(data: dict):
    for field in REQUIRED_NUMERIC:
        if field not in data:
            return f"Campo requerido faltante: '{field}'"
        if not isinstance(data[field], (int, float)):
            return f"'{field}' debe ser numérico"

    if data["MedInc"] <= 0:
        return "'MedInc' debe ser mayor que 0"
    if data["HouseAge"] <= 0:
        return "'HouseAge' debe ser mayor que 0"
    if data["AveRooms"] <= 0:
        return "'AveRooms' debe ser mayor que 0"
    if data["AveBedrms"] <= 0:
        return "'AveBedrms' debe ser mayor que 0"
    if data["Population"] <= 0:
        return "'Population' debe ser mayor que 0"
    if data["AveOccup"] <= 0:
        return "'AveOccup' debe ser mayor que 0"
    if data["Latitude"] < -90 or data["Latitude"] > 90:
        return "'Latitude' debe estar entre -90 y 90"
    if data["Longitude"] < -180 or data["Longitude"] > 180:
        return "'Longitude' debe estar entre -180 y 180"
    return None


# ── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "service":   "Housing Price Predictor",
        "version":   "1.0.0",
        "endpoints": ["/predict", "/health"],
    })

@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "train_metrics": metrics,
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Body debe ser JSON válido"}), 400

    error = validate(data)
    if error:
        return jsonify({"error": error}), 400

    try:
        features  = pd.DataFrame([data])
        log_price = pipeline.predict(features)[0]
        price     = float(np.expm1(log_price))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "predicted_price_usd": round(price, 2),
        "note": "Estimación basada en dataset California Housing ",
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Método HTTP no permitido en este endpoint"}), 405

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
