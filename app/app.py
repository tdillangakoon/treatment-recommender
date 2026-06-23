"""
app.py

Flask API for the Personalized Treatment Plan Recommender.

Endpoints:
    POST /predict    - patient symptoms in -> predicted condition category out
    POST /recommend  - condition category in -> recommended treatment out (saved to DB)
    POST /rate       - recommendation_id + doctor_rating in -> updates treatment alpha/beta

Run locally with:
    python app.py
Then the API is available at http://127.0.0.1:5000
"""

import pickle
import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

from database import get_connection, init_db

app = Flask(__name__, static_folder="static", static_url_path="")

# ---------------------------------------------------------------------------
# Load model + label encoder once at startup (not per-request - too slow)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "diagnosis_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "label_encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

FEATURE_ORDER = [
    "Age", "Fever_bin", "Cough_bin", "Fatigue_bin",
    "Difficulty Breathing_bin", "Gender_Male",
    "Blood_Pressure_enc", "Cholesterol_enc"
]

BP_MAP = {"Low": 0, "Normal": 1, "High": 2}
CHOL_MAP = {"Low": 0, "Normal": 1, "High": 2}

# Ensure the database + tables exist before the app starts handling requests
init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_feature_row(payload):
    """Convert a raw JSON patient payload into the encoded feature row the model expects."""
    row = {
        "Age": payload["age"],
        "Fever_bin": 1 if payload["fever"] else 0,
        "Cough_bin": 1 if payload["cough"] else 0,
        "Fatigue_bin": 1 if payload["fatigue"] else 0,
        "Difficulty Breathing_bin": 1 if payload["difficulty_breathing"] else 0,
        "Gender_Male": 1 if str(payload["gender"]).lower() == "male" else 0,
        "Blood_Pressure_enc": BP_MAP[payload["blood_pressure"]],
        "Cholesterol_enc": CHOL_MAP[payload["cholesterol_level"]],
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def sample_beta_recommendation(conn, category):
    """Thompson Sampling: pull treatments for this category from the DB and pick the best sample."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, treatment_name, alpha, beta FROM treatments WHERE condition_category = ?",
        (category,)
    )
    rows = cur.fetchall()
    if not rows:
        return None

    best_row = None
    best_sample = -1
    for row in rows:
        sample = np.random.beta(row["alpha"], row["beta"])
        if sample > best_sample:
            best_sample = sample
            best_row = row
    return best_row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
        "age": 30,
        "gender": "Female",
        "fever": true,
        "cough": true,
        "fatigue": true,
        "difficulty_breathing": true,
        "blood_pressure": "Normal",
        "cholesterol_level": "Normal"
    }
    """
    payload = request.get_json(force=True)

    required = ["age", "gender", "fever", "cough", "fatigue",
                "difficulty_breathing", "blood_pressure", "cholesterol_level"]
    missing = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = build_feature_row(payload)
    except KeyError as e:
        return jsonify({"error": f"Invalid value for field: {e}"}), 400

    pred_encoded = model.predict(X)[0]
    predicted_category = label_encoder.inverse_transform([pred_encoded])[0]

    # Save the patient record
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO patients
            (age, gender, fever, cough, fatigue, difficulty_breathing,
             blood_pressure, cholesterol_level, predicted_category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["age"], payload["gender"],
        int(payload["fever"]), int(payload["cough"]),
        int(payload["fatigue"]), int(payload["difficulty_breathing"]),
        payload["blood_pressure"], payload["cholesterol_level"],
        predicted_category
    ))
    patient_id = cur.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "patient_id": patient_id,
        "predicted_category": predicted_category
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Expected JSON body:
    {
        "patient_id": 1,
        "condition_category": "Respiratory"
    }
    """
    payload = request.get_json(force=True)

    if "patient_id" not in payload or "condition_category" not in payload:
        return jsonify({"error": "Missing patient_id or condition_category"}), 400

    category = payload["condition_category"]

    conn = get_connection()
    best_treatment = sample_beta_recommendation(conn, category)

    if best_treatment is None:
        conn.close()
        return jsonify({"error": f"Unknown condition_category: {category}"}), 400

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO recommendations (patient_id, treatment_id)
        VALUES (?, ?)
    """, (payload["patient_id"], best_treatment["id"]))
    recommendation_id = cur.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "recommendation_id": recommendation_id,
        "treatment_id": best_treatment["id"],
        "treatment_name": best_treatment["treatment_name"],
        "condition_category": category
    })


@app.route("/rate", methods=["POST"])
def rate():
    """
    Expected JSON body:
    {
        "recommendation_id": 1,
        "doctor_rating": 5
    }
    """
    payload = request.get_json(force=True)

    if "recommendation_id" not in payload or "doctor_rating" not in payload:
        return jsonify({"error": "Missing recommendation_id or doctor_rating"}), 400

    rating = payload["doctor_rating"]
    if not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({"error": "doctor_rating must be an integer between 1 and 5"}), 400

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT treatment_id FROM recommendations WHERE id = ?",
        (payload["recommendation_id"],)
    )
    rec_row = cur.fetchone()
    if rec_row is None:
        conn.close()
        return jsonify({"error": "Unknown recommendation_id"}), 404

    treatment_id = rec_row["treatment_id"]

    # Log the rating
    cur.execute("""
        INSERT INTO ratings (recommendation_id, doctor_rating)
        VALUES (?, ?)
    """, (payload["recommendation_id"], rating))

    # Update the bandit: success (alpha+1) if rating >= 4, else failure (beta+1)
    if rating >= 4:
        cur.execute("UPDATE treatments SET alpha = alpha + 1 WHERE id = ?", (treatment_id,))
    else:
        cur.execute("UPDATE treatments SET beta = beta + 1 WHERE id = ?", (treatment_id,))

    conn.commit()

    cur.execute("SELECT alpha, beta FROM treatments WHERE id = ?", (treatment_id,))
    updated = cur.fetchone()
    conn.close()

    return jsonify({
        "recommendation_id": payload["recommendation_id"],
        "treatment_id": treatment_id,
        "doctor_rating": rating,
        "updated_alpha": updated["alpha"],
        "updated_beta": updated["beta"]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
