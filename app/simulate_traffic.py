"""
simulate_traffic.py

Generates realistic simulated patient visits by calling the REAL Flask app
(/predict, /recommend, /rate) - so the data populating the database comes from
your actual classifier and actual bandit, not a separate fake generator.

Run this from inside the app/ folder, with the Flask server NOT running
(this script uses Flask's test client directly, it doesn't need the server up):

    python simulate_traffic.py

Safe to re-run - each run adds more simulated patients on top of existing data.
"""

import random
import time

import app as flask_app

client = flask_app.app.test_client()

N_PATIENTS = 150

# Rough symptom profiles per category - used only to make the simulated
# patients *plausible* inputs to the classifier, not to control its output.
# The classifier decides the actual predicted_category itself.
CATEGORY_PROFILES = {
    'Respiratory':         dict(fever=0.6, cough=0.7, fatigue=0.6, breathing=0.6, age=(15, 80)),
    'Gastrointestinal':    dict(fever=0.4, cough=0.2, fatigue=0.6, breathing=0.1, age=(15, 75)),
    'Cardiovascular':      dict(fever=0.2, cough=0.2, fatigue=0.6, breathing=0.3, age=(40, 90)),
    'Neurological':        dict(fever=0.2, cough=0.1, fatigue=0.6, breathing=0.1, age=(20, 85)),
    'Musculoskeletal':     dict(fever=0.1, cough=0.1, fatigue=0.5, breathing=0.1, age=(20, 80)),
    'Endocrine':           dict(fever=0.2, cough=0.1, fatigue=0.7, breathing=0.1, age=(25, 80)),
    'Infectious Disease':  dict(fever=0.7, cough=0.4, fatigue=0.5, breathing=0.2, age=(5, 70)),
    'Cancer':              dict(fever=0.3, cough=0.2, fatigue=0.8, breathing=0.2, age=(40, 85)),
    'Mental Health':       dict(fever=0.0, cough=0.0, fatigue=0.6, breathing=0.1, age=(15, 60)),
    'Renal/Urinary':       dict(fever=0.4, cough=0.0, fatigue=0.4, breathing=0.1, age=(20, 80)),
    'Dermatological':      dict(fever=0.0, cough=0.0, fatigue=0.2, breathing=0.0, age=(10, 70)),
    'Genetic/Congenital':  dict(fever=0.0, cough=0.0, fatigue=0.5, breathing=0.0, age=(1, 40)),
}

CATEGORIES = list(CATEGORY_PROFILES.keys())

# Secretly bias which treatment "works better" per category, so ratings aren't
# pure noise and the bandit has something real to learn from in the dashboard.
# (first treatment in each category list = the "good" one most of the time)
def biased_rating(category, treatment_name, treatments_for_cat):
    """Return a plausible doctor rating: the first-listed treatment per category
    skews toward higher ratings, others skew lower - simulates a bandit that
    actually has something to learn."""
    is_preferred = (treatment_name == treatments_for_cat[0])
    if is_preferred:
        return random.choices([3, 4, 5], weights=[1, 3, 4])[0]
    else:
        return random.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]


def run_simulation(n=N_PATIENTS):
    created = 0
    rated = 0

    for i in range(n):
        category_hint = random.choice(CATEGORIES)
        profile = CATEGORY_PROFILES[category_hint]

        age = random.randint(*profile['age'])
        gender = random.choice(['Male', 'Female'])
        fever = random.random() < profile['fever']
        cough = random.random() < profile['cough']
        fatigue = random.random() < profile['fatigue']
        breathing = random.random() < profile['breathing']
        bp = random.choices(['Low', 'Normal', 'High'], weights=[1, 6, 3])[0]
        chol = random.choices(['Low', 'Normal', 'High'], weights=[1, 6, 3])[0]

        resp = client.post('/predict', json={
            "age": age, "gender": gender,
            "fever": fever, "cough": cough, "fatigue": fatigue,
            "difficulty_breathing": breathing,
            "blood_pressure": bp, "cholesterol_level": chol
        })
        if resp.status_code != 200:
            continue
        data = resp.get_json()
        patient_id = data["patient_id"]
        predicted_category = data["predicted_category"]
        created += 1

        # Get a recommendation for the predicted category
        resp2 = client.post('/recommend', json={
            "patient_id": patient_id,
            "condition_category": predicted_category
        })
        if resp2.status_code != 200:
            continue
        rec_data = resp2.get_json()

        # Most (not all) recommendations get rated by a doctor - mirrors real usage
        if random.random() < 0.85:
            treatments_for_cat = flask_app.__dict__.get(
                'TREATMENTS_BY_CATEGORY', None
            )
            from database import TREATMENTS_BY_CATEGORY
            rating = biased_rating(
                predicted_category,
                rec_data["treatment_name"],
                TREATMENTS_BY_CATEGORY[predicted_category]
            )
            resp3 = client.post('/rate', json={
                "recommendation_id": rec_data["recommendation_id"],
                "doctor_rating": rating
            })
            if resp3.status_code == 200:
                rated += 1

    print(f"Simulation complete: {created} patients created, {rated} ratings submitted.")


if __name__ == "__main__":
    random.seed()  # different each run; remove/set a fixed seed if you want reproducibility
    run_simulation(N_PATIENTS)
