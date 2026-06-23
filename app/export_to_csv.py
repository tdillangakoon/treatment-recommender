"""
export_to_csv.py

Exports the treatments.db SQLite tables to CSV files for Power BI.
Power BI can import CSV natively (Get Data -> Text/CSV), so this avoids
needing to install an ODBC driver.

Produces a denormalized "dashboard-ready" CSV plus the 4 raw tables,
all written to a new powerbi_data/ folder.

Run from inside app/:
    python export_to_csv.py

Safe to re-run any time - just re-exports the latest data.
"""

import sqlite3
import csv
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "treatments.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "powerbi_data")


def export_table(conn, table_name, out_path):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"  {table_name}: {len(rows)} rows -> {out_path}")


def export_dashboard_view(conn, out_path):
    """
    A single denormalized table joining patients -> recommendations -> treatments -> ratings.
    This is the easiest one to build visuals from in Power BI, since it avoids
    needing to set up relationships between multiple tables.
    """
    query = """
        SELECT
            p.id AS patient_id,
            p.age,
            p.gender,
            p.fever,
            p.cough,
            p.fatigue,
            p.difficulty_breathing,
            p.blood_pressure,
            p.cholesterol_level,
            p.predicted_category,
            p.created_at AS patient_created_at,
            r.id AS recommendation_id,
            t.treatment_name,
            t.alpha AS treatment_current_alpha,
            t.beta AS treatment_current_beta,
            ROUND(CAST(t.alpha AS FLOAT) / (t.alpha + t.beta), 3) AS treatment_current_success_rate,
            rt.doctor_rating,
            rt.created_at AS rating_created_at
        FROM patients p
        LEFT JOIN recommendations r ON r.patient_id = p.id
        LEFT JOIN treatments t ON t.id = r.treatment_id
        LEFT JOIN ratings rt ON rt.recommendation_id = r.id
        ORDER BY p.id
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"  dashboard_view: {len(rows)} rows -> {out_path}")


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run database.py / the Flask app first.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    print("Exporting raw tables:")
    for table in ["patients", "treatments", "recommendations", "ratings"]:
        export_table(conn, table, os.path.join(OUT_DIR, f"{table}.csv"))

    print("\nExporting denormalized dashboard view:")
    export_dashboard_view(conn, os.path.join(OUT_DIR, "dashboard_view.csv"))

    conn.close()
    print(f"\nDone. CSV files are in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
