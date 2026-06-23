"""
database.py

SQLite database setup for the Personalized Treatment Plan Recommender.

Tables:
    patients         - patient profile + predicted condition category
    treatments       - treatment options per category, with live alpha/beta (bandit state)
    recommendations  - links a patient to a recommended treatment
    ratings          - doctor's rating (1-5) for a given recommendation

Run this file directly to create and seed treatments.db:
    python database.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "treatments.db")

# Same treatment set used in bandit.py - keep these in sync
TREATMENTS_BY_CATEGORY = {
    'Respiratory': ['Inhaled Bronchodilator', 'Antibiotic Course', 'Rest + Hydration Protocol'],
    'Gastrointestinal': ['Dietary Modification', 'Antacid/PPI Therapy', 'Antibiotic Course'],
    'Cardiovascular': ['ACE Inhibitor', 'Lifestyle + Statin Therapy', 'Beta Blocker'],
    'Neurological': ['Neurology Referral + MRI', 'Symptom Management Meds', 'Physical Therapy'],
    'Musculoskeletal': ['NSAID + Rest', 'Physical Therapy', 'Corticosteroid Injection'],
    'Endocrine': ['Hormone Therapy', 'Dietary + Insulin Management', 'Endocrinology Referral'],
    'Infectious Disease': ['Antibiotic Course', 'Antiviral Therapy', 'Supportive Care + Isolation'],
    'Cancer': ['Oncology Referral', 'Chemotherapy Protocol', 'Surgical Consultation'],
    'Mental Health': ['Psychotherapy Referral', 'SSRI Medication', 'Combined Therapy + Medication'],
    'Renal/Urinary': ['Antibiotic Course', 'Increased Hydration + Monitoring', 'Nephrology Referral'],
    'Dermatological': ['Topical Steroid', 'Antihistamine', 'Dermatology Referral'],
    'Genetic/Congenital': ['Specialist Referral', 'Supportive Care Plan', 'Genetic Counseling'],
}


def get_connection(db_path=DB_PATH):
    """Return a sqlite3 connection with foreign keys enforced and row access by column name."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            fever INTEGER NOT NULL DEFAULT 0,
            cough INTEGER NOT NULL DEFAULT 0,
            fatigue INTEGER NOT NULL DEFAULT 0,
            difficulty_breathing INTEGER NOT NULL DEFAULT 0,
            blood_pressure TEXT NOT NULL,
            cholesterol_level TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_category TEXT NOT NULL,
            treatment_name TEXT NOT NULL,
            alpha INTEGER NOT NULL DEFAULT 1,
            beta INTEGER NOT NULL DEFAULT 1,
            UNIQUE(condition_category, treatment_name)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            treatment_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (treatment_id) REFERENCES treatments(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id INTEGER NOT NULL,
            doctor_rating INTEGER NOT NULL CHECK (doctor_rating BETWEEN 1 AND 5),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
        )
    """)

    conn.commit()


def seed_treatments(conn):
    """Insert the default treatment set with alpha=1, beta=1 if not already present."""
    cur = conn.cursor()
    for category, treatments in TREATMENTS_BY_CATEGORY.items():
        for treatment_name in treatments:
            cur.execute("""
                INSERT OR IGNORE INTO treatments (condition_category, treatment_name, alpha, beta)
                VALUES (?, ?, 1, 1)
            """, (category, treatment_name))
    conn.commit()


def init_db(db_path=DB_PATH):
    """Create tables and seed treatments. Safe to run multiple times."""
    conn = get_connection(db_path)
    create_tables(conn)
    seed_treatments(conn)
    conn.close()
    print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    init_db()

    # Quick verification
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as count FROM treatments")
    print("Total treatments seeded:", cur.fetchone()["count"])

    cur.execute("SELECT * FROM treatments WHERE condition_category = ?", ("Respiratory",))
    print("\nRespiratory treatments:")
    for row in cur.fetchall():
        print(dict(row))

    conn.close()
