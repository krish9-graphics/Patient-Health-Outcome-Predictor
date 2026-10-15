from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import sqlite3
from datetime import datetime


# ==========================================
# FLASK APP CONFIGURATION
# ==========================================

app = Flask(__name__)
CORS(app)


# ==========================================
# LOAD ML MODEL AND PREPROCESSOR
# ==========================================

model = joblib.load("model/logistic_regression_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DATABASE = "database.db"


def create_database():
    """
    Creates the SQLite database and patient_records
    table if they do not already exist.
    """

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            Age INTEGER,
            Gender TEXT,
            BMI REAL,
            Smoking_Status TEXT,
            Alcohol_Use TEXT,
            Physical_Activity TEXT,
            Family_History TEXT,

            Diabetes TEXT,
            Hypertension TEXT,
            Heart_Disease TEXT,
            Kidney_Disease TEXT,
            Respiratory_Disease TEXT,

            Systolic_BP REAL,
            Diastolic_BP REAL,
            Glucose_mg_dL REAL,
            Cholesterol_mg_dL REAL,
            Heart_Rate_bpm REAL,

            Disease_Category TEXT,
            Treatment_Plan TEXT,
            Treatment_Duration_Days INTEGER,
            Treatment_Adherence_Percent REAL,
            Medication_Count INTEGER,
            Previous_Hospitalizations INTEGER,
            Followup_Visits INTEGER,
            Days_to_Followup INTEGER,
            Side_Effects TEXT,

            prediction TEXT,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


# Create database and table when Flask starts
create_database()


# ==========================================
# HOME / HEALTH CHECK ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Patient Health Outcome Predictor API is running"
    })


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------
        # 1. RECEIVE DATA FROM REACT
        # --------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No patient data received"
            }), 400


        # --------------------------------------
        # 2. CONVERT JSON TO DATAFRAME
        # --------------------------------------

        patient_data = pd.DataFrame([data])


        # --------------------------------------
        # 3. PREPROCESS PATIENT DATA
        # --------------------------------------

        processed_data = preprocessor.transform(patient_data)


        # --------------------------------------
        # 4. MAKE ML PREDICTION
        # --------------------------------------

        prediction = model.predict(processed_data)[0]


        # --------------------------------------
        # 5. SAVE PATIENT RECORD TO DATABASE
        # --------------------------------------

        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO patient_records (

                Age,
                Gender,
                BMI,
                Smoking_Status,
                Alcohol_Use,
                Physical_Activity,
                Family_History,

                Diabetes,
                Hypertension,
                Heart_Disease,
                Kidney_Disease,
                Respiratory_Disease,

                Systolic_BP,
                Diastolic_BP,
                Glucose_mg_dL,
                Cholesterol_mg_dL,
                Heart_Rate_bpm,

                Disease_Category,
                Treatment_Plan,
                Treatment_Duration_Days,
                Treatment_Adherence_Percent,
                Medication_Count,
                Previous_Hospitalizations,
                Followup_Visits,
                Days_to_Followup,
                Side_Effects,

                prediction,
                created_at
            )

            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
        """, (

            data["Age"],
            data["Gender"],
            data["BMI"],
            data["Smoking_Status"],
            data["Alcohol_Use"],
            data["Physical_Activity"],
            data["Family_History"],

            data["Diabetes"],
            data["Hypertension"],
            data["Heart_Disease"],
            data["Kidney_Disease"],
            data["Respiratory_Disease"],

            data["Systolic_BP"],
            data["Diastolic_BP"],
            data["Glucose_mg_dL"],
            data["Cholesterol_mg_dL"],
            data["Heart_Rate_bpm"],

            data["Disease_Category"],
            data["Treatment_Plan"],
            data["Treatment_Duration_Days"],
            data["Treatment_Adherence_Percent"],
            data["Medication_Count"],
            data["Previous_Hospitalizations"],
            data["Followup_Visits"],
            data["Days_to_Followup"],
            data["Side_Effects"],

            prediction,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        connection.commit()
        connection.close()


        # --------------------------------------
        # 6. RETURN PREDICTION TO REACT
        # --------------------------------------

        return jsonify({
            "prediction": prediction
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# GET ALL PATIENT RECORDS
# ==========================================

@app.route("/patients", methods=["GET"])
def get_patients():

    try:

        # Connect to database
        connection = sqlite3.connect(DATABASE)

        # Return database rows as dictionaries
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()


        # Get all records
        cursor.execute("""
            SELECT *
            FROM patient_records
            ORDER BY id DESC
        """)

        records = cursor.fetchall()

        connection.close()


        # Convert SQLite rows to Python dictionaries
        patients = [dict(record) for record in records]


        return jsonify(patients)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# START FLASK SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )