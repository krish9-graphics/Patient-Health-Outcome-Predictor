# Theme/UI fixes: sidebar remains white text in both themes; Streamlit toolbar hidden.
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Patient Health Outcome Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "backend" / "database.db"

MODEL_PATH = (
    BASE_DIR
    / "backend"
    / "model"
    / "logistic_regression_model.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "backend"
    / "model"
    / "preprocessor.pkl"
)

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "patient_health_outcome_predictor_30000.csv"
)


# ============================================================
# ============================================================
# THEME MODE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Streamlit reruns the script automatically when the toggle changes.
dark_mode = st.session_state["dark_mode"]


# ============================================================
# THEME COLORS
# ============================================================

if dark_mode:
    # Professional dark healthcare dashboard palette
    BG_COLOR = "#0B1220"
    SURFACE_COLOR = "#111827"
    SURFACE_ALT_COLOR = "#172033"
    SIDEBAR_COLOR = "#0F172A"
    BORDER_COLOR = "#263244"
    TEXT_COLOR = "#F8FAFC"
    MUTED_COLOR = "#94A3B8"
    PRIMARY_COLOR = "#60A5FA"
    PRIMARY_HOVER = "#3B82F6"
    INPUT_COLOR = "#111827"
    INPUT_BORDER = "#334155"
    INFO_BG = "#102A43"
    INFO_BORDER = "#2563EB"
    PLOTLY_TEMPLATE = "plotly_dark"
else:
    # Clean light healthcare dashboard palette
    BG_COLOR = "#F5F7FB"
    SURFACE_COLOR = "#FFFFFF"
    SURFACE_ALT_COLOR = "#F8FAFC"
    SIDEBAR_COLOR = "#0F172A"
    BORDER_COLOR = "#E2E8F0"
    TEXT_COLOR = "#172033"
    MUTED_COLOR = "#667085"
    PRIMARY_COLOR = "#4F46E5"
    PRIMARY_HOVER = "#4338CA"
    INPUT_COLOR = "#FFFFFF"
    INPUT_BORDER = "#CBD5E1"
    INFO_BG = "#EFF6FF"
    INFO_BORDER = "#BFDBFE"
    PLOTLY_TEMPLATE = "plotly_white"


# Plotly defaults are changed before any chart is created.
px.defaults.template = PLOTLY_TEMPLATE
px.defaults.color_discrete_sequence = [
    "#4F46E5",
    "#0EA5E9",
    "#10B981",
    "#F59E0B",
    "#EF4444",
    "#8B5CF6",
    "#14B8A6"
]


# ============================================================
# PLOTLY THEME HELPER
# ============================================================

def style_plotly(fig):
    """Apply the current application theme to a Plotly figure."""

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font=dict(
            color=TEXT_COLOR,
            family="Inter, Segoe UI, Arial, sans-serif"
        ),
        title_font=dict(
            color=TEXT_COLOR,
            size=16
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR)
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    fig.update_xaxes(
        title_font=dict(color=MUTED_COLOR),
        tickfont=dict(color=MUTED_COLOR),
        gridcolor=BORDER_COLOR,
        zerolinecolor=BORDER_COLOR
    )

    fig.update_yaxes(
        title_font=dict(color=MUTED_COLOR),
        tickfont=dict(color=MUTED_COLOR),
        gridcolor=BORDER_COLOR,
        zerolinecolor=BORDER_COLOR
    )

    return fig


# ============================================================
# CUSTOM CSS
# ============================================================

# The CSS uses the same color variables as the Python theme above.
theme_css = """
<style>

/* ============================================================
   APPLICATION BACKGROUND
   ============================================================ */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: BG_COLOR !important;
    color: TEXT_COLOR !important;
}

[data-testid="stHeader"] {
    background-color: BG_COLOR !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main text and headings */
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
p,
label,
[data-testid="stText"] {
    color: TEXT_COLOR;
}

h1, h2, h3, h4, h5, h6 {
    color: TEXT_COLOR !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: SIDEBAR_COLOR !important;
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}


/* ============================================================
   HEADERS / SECTIONS
   ============================================================ */

.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
    color: PRIMARY_COLOR !important;
}

.subtitle {
    color: MUTED_COLOR !important;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: PRIMARY_COLOR !important;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-description {
    color: MUTED_COLOR !important;
    margin-bottom: 18px;
}


/* ============================================================
   NATIVE STREAMLIT METRICS
   ============================================================ */

[data-testid="stMetric"] {
    background-color: SURFACE_COLOR !important;
    border: 1px solid BORDER_COLOR !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: MUTED_COLOR !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] span {
    color: TEXT_COLOR !important;
}

[data-testid="stMetricDelta"] {
    color: MUTED_COLOR !important;
}


/* ============================================================
   CUSTOM CARDS
   ============================================================ */

.metric-card {
    background-color: SURFACE_COLOR !important;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid BORDER_COLOR;
    min-height: 120px;
}

.metric-title {
    color: MUTED_COLOR !important;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    color: PRIMARY_COLOR !important;
    font-size: 28px;
    font-weight: 700;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    background-color: PRIMARY_COLOR !important;
    color: #FFFFFF !important;
    border: 1px solid PRIMARY_COLOR !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: PRIMARY_HOVER !important;
    border-color: PRIMARY_HOVER !important;
    color: #FFFFFF !important;
}

.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}


/* ============================================================
   INPUTS / NUMBER INPUTS
   ============================================================ */

div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
    background-color: INPUT_COLOR !important;
    border-color: INPUT_BORDER !important;
}

input,
textarea {
    background-color: INPUT_COLOR !important;
    color: TEXT_COLOR !important;
    caret-color: TEXT_COLOR !important;
}

input::placeholder,
textarea::placeholder {
    color: MUTED_COLOR !important;
}


/* ============================================================
   SELECTBOX / MULTI SELECT
   ============================================================ */

div[data-baseweb="select"] > div {
    background-color: INPUT_COLOR !important;
    border-color: INPUT_BORDER !important;
    color: TEXT_COLOR !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: TEXT_COLOR !important;
}

/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: SURFACE_COLOR !important;
    border-color: BORDER_COLOR !important;
}

li[role="option"] {
    background-color: SURFACE_COLOR !important;
    color: TEXT_COLOR !important;
}

li[role="option"]:hover {
    background-color: SURFACE_ALT_COLOR !important;
}


/* ============================================================
   SLIDER
   ============================================================ */

[data-testid="stSlider"] label,
[data-testid="stSlider"] p {
    color: TEXT_COLOR !important;
}


/* ============================================================
   RADIO / TOGGLE
   ============================================================ */

[data-testid="stRadio"] label,
[data-testid="stToggle"] label,
[data-testid="stRadio"] p,
[data-testid="stToggle"] p {
    color: TEXT_COLOR !important;
}


/* ============================================================
   DATAFRAME / TABLE
   ============================================================ */

[data-testid="stDataFrame"] {
    border: 1px solid BORDER_COLOR !important;
    border-radius: 10px !important;
}


/* ============================================================
   EXPANDERS
   ============================================================ */

[data-testid="stExpander"] {
    background-color: SURFACE_COLOR !important;
    border: 1px solid BORDER_COLOR !important;
    border-radius: 10px !important;
}


/* ============================================================
   INFO BOX / ALERTS
   ============================================================ */

.info-box {
    padding: 16px;
    border-radius: 10px;
    background-color: INFO_BG !important;
    border: 1px solid INFO_BORDER !important;
    margin-top: 10px;
    margin-bottom: 15px;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
}


/* ============================================================
   SIDEBAR — ALWAYS WHITE TEXT
   ============================================================ */

/* The sidebar stays dark blue in both themes, so its text must
   never inherit the light-theme main-content text color. */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] span,
section[data-testid="stSidebar"] [data-testid="stToggle"] label,
section[data-testid="stSidebar"] [data-testid="stToggle"] label p,
section[data-testid="stSidebar"] [data-testid="stToggle"] span {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}

/* ============================================================
   DIVIDERS / FOOTER
   ============================================================ */

hr {
    border-color: BORDER_COLOR !important;
}

.footer {
    text-align: center;
    color: MUTED_COLOR !important;
    font-size: 13px;
    padding-top: 40px;
}


/* ============================================================
   PREDICTION RESULT
   ============================================================ */

.prediction-result {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
    background-color: SURFACE_COLOR !important;
    border: 1px solid BORDER_COLOR !important;
}

.prediction-title {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 8px;
    color: MUTED_COLOR !important;
}

.prediction-value {
    font-size: 32px;
    font-weight: 700;
    color: PRIMARY_COLOR !important;
}

</style>
"""

for name, value in {
    "BG_COLOR": BG_COLOR,
    "SURFACE_COLOR": SURFACE_COLOR,
    "SURFACE_ALT_COLOR": SURFACE_ALT_COLOR,
    "SIDEBAR_COLOR": SIDEBAR_COLOR,
    "BORDER_COLOR": BORDER_COLOR,
    "TEXT_COLOR": TEXT_COLOR,
    "MUTED_COLOR": MUTED_COLOR,
    "PRIMARY_COLOR": PRIMARY_COLOR,
    "PRIMARY_HOVER": PRIMARY_HOVER,
    "INPUT_COLOR": INPUT_COLOR,
    "INPUT_BORDER": INPUT_BORDER,
    "INFO_BG": INFO_BG,
    "INFO_BORDER": INFO_BORDER,
}.items():
    theme_css = theme_css.replace(name, value)

st.markdown(theme_css, unsafe_allow_html=True)


# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


@st.cache_data(ttl=2)
def get_patient_records():
    """
    Read patient records from SQLite database.
    """

    try:

        conn = get_connection()

        query = """
        SELECT *
        FROM patient_records
        ORDER BY id ASC
        """

        df = pd.read_sql_query(query, conn)

        conn.close()

        return df

    except Exception as e:

        st.error(f"Unable to load patient records: {e}")

        return pd.DataFrame()


def save_patient_record(patient_data, prediction):

    conn = get_connection()

    cursor = conn.cursor()

    query = """
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
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime')
    )
    """

    values = (
        patient_data["Age"],
        patient_data["Gender"],
        patient_data["BMI"],
        patient_data["Smoking_Status"],
        patient_data["Alcohol_Use"],
        patient_data["Physical_Activity"],
        patient_data["Family_History"],
        patient_data["Diabetes"],
        patient_data["Hypertension"],
        patient_data["Heart_Disease"],
        patient_data["Kidney_Disease"],
        patient_data["Respiratory_Disease"],
        patient_data["Systolic_BP"],
        patient_data["Diastolic_BP"],
        patient_data["Glucose_mg_dL"],
        patient_data["Cholesterol_mg_dL"],
        patient_data["Heart_Rate_bpm"],
        patient_data["Disease_Category"],
        patient_data["Treatment_Plan"],
        patient_data["Treatment_Duration_Days"],
        patient_data["Treatment_Adherence_Percent"],
        patient_data["Medication_Count"],
        patient_data["Previous_Hospitalizations"],
        patient_data["Followup_Visits"],
        patient_data["Days_to_Followup"],
        patient_data["Side_Effects"],
        prediction
    )

    cursor.execute(query, values)

    conn.commit()

    conn.close()

    get_patient_records.clear()


# ============================================================
# DATASET OPTIONS
# ============================================================

def get_options(column):

    dataset = load_dataset()

    if column not in dataset.columns:
        return []

    values = (
        dataset[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


# ============================================================
# RANDOM SAMPLE
# ============================================================

def generate_random_sample():

    dataset = load_dataset()

    sample = dataset.sample(
        n=1
    ).iloc[0]

    patient = {}

    feature_columns = [
        "Age",
        "Gender",
        "BMI",
        "Smoking_Status",
        "Alcohol_Use",
        "Physical_Activity",
        "Family_History",
        "Diabetes",
        "Hypertension",
        "Heart_Disease",
        "Kidney_Disease",
        "Respiratory_Disease",
        "Systolic_BP",
        "Diastolic_BP",
        "Glucose_mg_dL",
        "Cholesterol_mg_dL",
        "Heart_Rate_bpm",
        "Disease_Category",
        "Treatment_Plan",
        "Treatment_Duration_Days",
        "Treatment_Adherence_Percent",
        "Medication_Count",
        "Previous_Hospitalizations",
        "Followup_Visits",
        "Days_to_Followup",
        "Side_Effects"
    ]

    for column in feature_columns:

        if column in dataset.columns:

            value = sample[column]

            if pd.isna(value):
                value = ""

            patient[column] = value

    return patient


# ============================================================
# MODEL PREDICTION
# ============================================================

def make_prediction(patient_data):

    model = load_model()

    preprocessor = load_preprocessor()

    input_df = pd.DataFrame(
        [patient_data]
    )

    transformed_data = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        transformed_data
    )[0]

    return str(prediction)


# ============================================================
# VALIDATION
# ============================================================

def validate_patient(data):

    errors = []

    # ----------------------------
    # Required fields
    # ----------------------------

    required_fields = [
        "Age",
        "BMI",
        "Gender",
        "Smoking_Status",
        "Alcohol_Use",
        "Physical_Activity",
        "Family_History",
        "Diabetes",
        "Hypertension",
        "Heart_Disease",
        "Kidney_Disease",
        "Respiratory_Disease",
        "Systolic_BP",
        "Diastolic_BP",
        "Glucose_mg_dL",
        "Cholesterol_mg_dL",
        "Heart_Rate_bpm",
        "Disease_Category",
        "Treatment_Plan",
        "Treatment_Duration_Days",
        "Treatment_Adherence_Percent",
        "Medication_Count",
        "Previous_Hospitalizations",
        "Followup_Visits",
        "Days_to_Followup",
        "Side_Effects"
    ]

    for field in required_fields:

        value = data.get(field)

        if value is None or value == "":

            errors.append(
                f"Please provide {field.replace('_', ' ')}."
            )


    # ----------------------------
    # Numeric validation
    # ----------------------------

    try:

        age = float(data["Age"])

        if age < 18 or age > 90:

            errors.append(
                "Age must be between 18 and 90."
            )

    except:
        pass


    try:

        bmi = float(data["BMI"])

        if bmi < 15 or bmi > 45:

            errors.append(
                "BMI must be between 15 and 45."
            )

    except:
        pass


    try:

        systolic = float(
            data["Systolic_BP"]
        )

        if systolic < 80 or systolic > 250:

            errors.append(
                "Systolic BP must be between 80 and 250 mmHg."
            )

    except:
        pass


    try:

        diastolic = float(
            data["Diastolic_BP"]
        )

        if diastolic < 40 or diastolic > 150:

            errors.append(
                "Diastolic BP must be between 40 and 150 mmHg."
            )

    except:
        pass


    try:

        systolic = float(
            data["Systolic_BP"]
        )

        diastolic = float(
            data["Diastolic_BP"]
        )

        if systolic <= diastolic:

            errors.append(
                "Systolic BP should be greater than diastolic BP."
            )

    except:
        pass


    try:

        glucose = float(
            data["Glucose_mg_dL"]
        )

        if glucose < 50 or glucose > 500:

            errors.append(
                "Glucose must be between 50 and 500 mg/dL."
            )

    except:
        pass


    try:

        cholesterol = float(
            data["Cholesterol_mg_dL"]
        )

        if cholesterol < 100 or cholesterol > 500:

            errors.append(
                "Cholesterol must be between 100 and 500 mg/dL."
            )

    except:
        pass


    try:

        heart_rate = float(
            data["Heart_Rate_bpm"]
        )

        if heart_rate < 30 or heart_rate > 220:

            errors.append(
                "Heart rate must be between 30 and 220 bpm."
            )

    except:
        pass


    try:

        duration = float(
            data["Treatment_Duration_Days"]
        )

        if duration < 1 or duration > 1000:

            errors.append(
                "Treatment duration must be between 1 and 1000 days."
            )

    except:
        pass


    try:

        adherence = float(
            data["Treatment_Adherence_Percent"]
        )

        if adherence < 25 or adherence > 100:

            errors.append(
                "Treatment adherence must be between 25% and 100%."
            )

    except:
        pass


    try:

        medications = float(
            data["Medication_Count"]
        )

        if medications < 0 or medications > 10:

            errors.append(
                "Medication count must be between 0 and 10."
            )

    except:
        pass


    try:

        hospitalizations = float(
            data["Previous_Hospitalizations"]
        )

        if hospitalizations < 0 or hospitalizations > 6:

            errors.append(
                "Previous hospitalizations must be between 0 and 6."
            )

    except:
        pass


    try:

        followups = float(
            data["Followup_Visits"]
        )

        if followups < 0 or followups > 7:

            errors.append(
                "Follow-up visits must be between 0 and 7."
            )

    except:
        pass


    try:

        days = float(
            data["Days_to_Followup"]
        )

        if days < 0 or days > 365:

            errors.append(
                "Days to follow-up must be between 0 and 365."
            )

    except:
        pass


    return errors


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:12px;
            font-weight:600;
            color:#94A3B8;
            text-transform:uppercase;
            letter-spacing:0.08em;
            margin-bottom:6px;
        ">
            Appearance
        </div>
        """,
        unsafe_allow_html=True
    )

    st.toggle(
        "🌙 Dark Mode",
        key="dark_mode"
    )

    st.markdown(
        "<div style=\"height:12px;\"></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:700;
            margin-bottom:5px;
        ">
            🏥 Patient Health
        </div>

        <div style="
            font-size:13px;
            color:#9ca3af;
            margin-bottom:25px;
        ">
            Outcome Predictor
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "➕ New Prediction",
            "📋 Patient History",
            "🦠 Disease Analysis",
            "💊 Treatment Analysis",
            "⚠️ Risk Analysis"
        ]
    )

    st.markdown("---")

    # st.caption(
    #     "AI-powered patient outcome prediction"
    # )


# ============================================================
# LOAD RECORDS
# ============================================================

patients = get_patient_records()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Overview of patient predictions and health outcome statistics.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "➕ New Prediction"
    ):

        st.session_state["page"] = "➕ New Prediction"

        st.rerun()


    # ----------------------------------------
    # Basic statistics
    # ----------------------------------------

    total_patients = len(patients)

    improved = 0
    stable = 0
    deteriorated = 0

    if total_patients > 0:

        improved = (
            patients["prediction"]
            .astype(str)
            .str.lower()
            .eq("improved")
            .sum()
        )

        stable = (
            patients["prediction"]
            .astype(str)
            .str.lower()
            .eq("stable")
            .sum()
        )

        deteriorated = (
            patients["prediction"]
            .astype(str)
            .str.lower()
            .eq("deteriorated")
            .sum()
        )


    # ----------------------------------------
    # Metric cards
    # ----------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Patients",
        total_patients
    )

    col2.metric(
        "Improved",
        improved
    )

    col3.metric(
        "Stable",
        stable
    )

    col4.metric(
        "Deteriorated",
        deteriorated
    )


    st.markdown(
        '<div class="section-title">Patient Statistics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Average values from submitted patient records.</div>',
        unsafe_allow_html=True
    )


    # ----------------------------------------
    # Average statistics
    # ----------------------------------------

    if total_patients > 0:

        average_age = pd.to_numeric(
            patients["Age"],
            errors="coerce"
        ).mean()

        average_bmi = pd.to_numeric(
            patients["BMI"],
            errors="coerce"
        ).mean()

        average_adherence = pd.to_numeric(
            patients["Treatment_Adherence_Percent"],
            errors="coerce"
        ).mean()

        average_hospitalizations = pd.to_numeric(
            patients["Previous_Hospitalizations"],
            errors="coerce"
        ).mean()

    else:

        average_age = 0
        average_bmi = 0
        average_adherence = 0
        average_hospitalizations = 0


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Age",
        f"{average_age:.1f} years"
    )

    c2.metric(
        "Average BMI",
        f"{average_bmi:.1f}"
    )

    c3.metric(
        "Average Treatment Adherence",
        f"{average_adherence:.1f}%"
    )

    c4.metric(
        "Avg. Hospitalizations",
        f"{average_hospitalizations:.1f}"
    )


    # ----------------------------------------
    # Outcome Distribution
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">Outcome Distribution</div>',
        unsafe_allow_html=True
    )

    if total_patients == 0:

        st.info(
            "No prediction data available."
        )

    else:

        outcome_df = pd.DataFrame(
            {
                "Outcome": [
                    "Improved",
                    "Stable",
                    "Deteriorated"
                ],
                "Patients": [
                    improved,
                    stable,
                    deteriorated
                ]
            }
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                outcome_df,
                x="Outcome",
                y="Patients",
                title="Predicted Patient Outcomes"
            )

            fig.update_layout(
                showlegend=False,
                height=350
            )

            st.plotly_chart(
                style_plotly(fig),
                use_container_width=True
            )

        with col2:

            fig = px.pie(
                outcome_df,
                names="Outcome",
                values="Patients",
                title="Outcome Percentage"
            )

            fig.update_layout(
                height=350
            )

            st.plotly_chart(
                style_plotly(fig),
                use_container_width=True
            )


    # ----------------------------------------
    # Recent predictions
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">Recent Predictions</div>',
        unsafe_allow_html=True
    )

    if total_patients == 0:

        st.info(
            "No patient records available."
        )

    else:

        recent_patients = (
            patients
            .sort_values(
                "id",
                ascending=False
            )
            .head(5)
            .sort_values(
                "id",
                ascending=True
            )
        )

        recent_display = recent_patients[
            [
                "id",
                "Age",
                "Disease_Category",
                "prediction",
                "created_at"
            ]
        ].copy()

        recent_display.columns = [
            "Patient ID",
            "Age",
            "Disease",
            "Prediction",
            "Date"
        ]

        st.dataframe(
            recent_display,
            use_container_width=True,
            hide_index=True
        )


    # ----------------------------------------
    # Disease distribution
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">Disease Distribution</div>',
        unsafe_allow_html=True
    )

    if total_patients > 0:

        disease_counts = (
            patients["Disease_Category"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        disease_counts.columns = [
            "Disease",
            "Patients"
        ]

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                disease_counts,
                x="Disease",
                y="Patients",
                title="Patients by Disease Category"
            )

            st.plotly_chart(
                style_plotly(fig),
                use_container_width=True
            )

        with col2:

            fig = px.pie(
                disease_counts,
                names="Disease",
                values="Patients",
                title="Disease Percentage"
            )

            st.plotly_chart(
                style_plotly(fig),
                use_container_width=True
            )

    else:

        st.info(
            "No disease data available."
        )


    # ----------------------------------------
    # Heart Disease overview
    # ----------------------------------------

    st.markdown(
        '<div class="section-title">Heart Disease Overview</div>',
        unsafe_allow_html=True
    )

    if total_patients > 0:

        heart_patients = patients[
            patients["Heart_Disease"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                ["yes", "1", "true"]
            )
        ]

        heart_count = len(
            heart_patients
        )

        st.metric(
            "Patients with Heart Disease",
            heart_count
        )

        if heart_count > 0:

            heart_outcomes = (
                heart_patients["prediction"]
                .value_counts()
                .reset_index()
            )

            heart_outcomes.columns = [
                "Outcome",
                "Patients"
            ]

            fig = px.bar(
                heart_outcomes,
                x="Outcome",
                y="Patients",
                title="Predicted Outcomes Among Heart Disease Patients"
            )

            st.plotly_chart(
                style_plotly(fig),
                use_container_width=True
            )


# ============================================================
# NEW PREDICTION
# ============================================================

elif page == "➕ New Prediction":

    st.markdown(
        '<div class="main-title">New Patient Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Enter the patient's information below.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SAMPLE / CLEAR BUTTONS
    # ========================================================

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        if st.button(
            "🎲 Load Random Sample",
            use_container_width=True
        ):

            st.session_state["sample_patient"] = (
                generate_random_sample()
            )

            st.session_state["prediction"] = None

            st.rerun()


    with col2:

        if st.button(
            "🗑️ Clear Form",
            use_container_width=True
        ):

            st.session_state.pop(
                "sample_patient",
                None
            )

            st.session_state.pop(
                "prediction",
                None
            )

            st.rerun()


    sample = st.session_state.get(
        "sample_patient",
        {}
    )


    # ========================================================
    # SECTION 01 — DEMOGRAPHICS
    # ========================================================

    st.markdown(
        '<div class="section-title">01 — Demographics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Basic information about the patient.</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=90,
            value=int(
                float(sample.get("Age", 45))
            ),
            step=1
        )


    with col2:

        gender_options = [
            "Male",
            "Female",
            "Other"
        ]

        gender_default = sample.get(
            "Gender",
            "Male"
        )

        gender_index = (
            gender_options.index(
                gender_default
            )
            if gender_default in gender_options
            else 0
        )

        gender = st.selectbox(
            "Gender",
            gender_options,
            index=gender_index
        )


    with col3:

        bmi = st.number_input(
            "BMI",
            min_value=15.0,
            max_value=45.0,
            value=float(
                sample.get("BMI", 25.0)
            ),
            step=0.1
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        smoking_options = [
            "Never",
            "Former",
            "Current"
        ]

        smoking_default = sample.get(
            "Smoking_Status",
            "Never"
        )

        smoking_index = (
            smoking_options.index(
                smoking_default
            )
            if smoking_default in smoking_options
            else 0
        )

        smoking_status = st.selectbox(
            "Smoking Status",
            smoking_options,
            index=smoking_index
        )


    with col2:

        alcohol_options = [
            "Occasional",
            "Regular",
            "Unknown"
        ]

        alcohol_default = sample.get(
            "Alcohol_Use",
            "Unknown"
        )

        alcohol_index = (
            alcohol_options.index(
                alcohol_default
            )
            if alcohol_default in alcohol_options
            else 0
        )

        alcohol_use = st.selectbox(
            "Alcohol Use",
            alcohol_options,
            index=alcohol_index
        )


    with col3:

        activity_options = [
            "Low",
            "Moderate",
            "High"
        ]

        activity_default = sample.get(
            "Physical_Activity",
            "Moderate"
        )

        activity_index = (
            activity_options.index(
                activity_default
            )
            if activity_default in activity_options
            else 1
        )

        physical_activity = st.selectbox(
            "Physical Activity",
            activity_options,
            index=activity_index
        )


    # ========================================================
    # SECTION 02 — MEDICAL HISTORY
    # ========================================================

    st.markdown(
        '<div class="section-title">02 — Medical History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Existing conditions and medical background.</div>',
        unsafe_allow_html=True
    )


    medical_fields = [
        ("Family_History", "Family History"),
        ("Diabetes", "Diabetes"),
        ("Hypertension", "Hypertension"),
        ("Heart_Disease", "Heart Disease"),
        ("Kidney_Disease", "Kidney Disease"),
        ("Respiratory_Disease", "Respiratory Disease")
    ]


    medical_values = {}


    cols = st.columns(3)

    for i, (field, label) in enumerate(
        medical_fields
    ):

        with cols[i % 3]:

            default_value = sample.get(
                field,
                "No"
            )

            options = [
                "Yes",
                "No"
            ]

            index = (
                options.index(
                    default_value
                )
                if default_value in options
                else 1
            )

            medical_values[field] = st.selectbox(
                label,
                options,
                index=index,
                key=f"medical_{field}"
            )


    # ========================================================
    # SECTION 03 — CLINICAL MEASUREMENTS
    # ========================================================

    st.markdown(
        '<div class="section-title">03 — Clinical Measurements</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Current clinical and vital measurements.</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)

    with col1:

        systolic_bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=80,
            max_value=250,
            value=int(
                float(
                    sample.get(
                        "Systolic_BP",
                        120
                    )
                )
            ),
            step=1
        )


    with col2:

        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=40,
            max_value=150,
            value=int(
                float(
                    sample.get(
                        "Diastolic_BP",
                        80
                    )
                )
            ),
            step=1
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=50,
            max_value=500,
            value=int(
                float(
                    sample.get(
                        "Glucose_mg_dL",
                        120
                    )
                )
            ),
            step=1
        )


    with col2:

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=500,
            value=int(
                float(
                    sample.get(
                        "Cholesterol_mg_dL",
                        200
                    )
                )
            ),
            step=1
        )


    with col3:

        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=30,
            max_value=220,
            value=int(
                float(
                    sample.get(
                        "Heart_Rate_bpm",
                        75
                    )
                )
            ),
            step=1
        )


    disease_options = [
        "Diabetes",
        "Cardiovascular",
        "Respiratory",
        "Kidney",
        "Other"
    ]

    disease_default = sample.get(
        "Disease_Category",
        "Other"
    )

    disease_index = (
        disease_options.index(
            disease_default
        )
        if disease_default in disease_options
        else 0
    )

    disease_category = st.selectbox(
        "Disease Category",
        disease_options,
        index=disease_index
    )


    # ========================================================
    # SECTION 04 — TREATMENT
    # ========================================================

    st.markdown(
        '<div class="section-title">04 — Treatment Details</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Current treatment and patient adherence.</div>',
        unsafe_allow_html=True
    )


    treatment_options = [
        "Medication",
        "Surgery",
        "Therapy",
        "Lifestyle",
        "Medication + Therapy"
    ]

    treatment_default = sample.get(
        "Treatment_Plan",
        "Medication"
    )

    treatment_index = (
        treatment_options.index(
            treatment_default
        )
        if treatment_default in treatment_options
        else 0
    )


    treatment_plan = st.selectbox(
        "Treatment Plan",
        treatment_options,
        index=treatment_index
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        treatment_duration = st.number_input(
            "Treatment Duration (Days)",
            min_value=1,
            max_value=1000,
            value=int(
                float(
                    sample.get(
                        "Treatment_Duration_Days",
                        60
                    )
                )
            ),
            step=1
        )


    with col2:

        medication_count = st.number_input(
            "Medication Count",
            min_value=0,
            max_value=10,
            value=int(
                float(
                    sample.get(
                        "Medication_Count",
                        2
                    )
                )
            ),
            step=1
        )


    with col3:

        previous_hospitalizations = st.number_input(
            "Previous Hospitalizations",
            min_value=0,
            max_value=6,
            value=int(
                float(
                    sample.get(
                        "Previous_Hospitalizations",
                        1
                    )
                )
            ),
            step=1
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        followup_visits = st.number_input(
            "Follow-up Visits",
            min_value=0,
            max_value=7,
            value=int(
                float(
                    sample.get(
                        "Followup_Visits",
                        2
                    )
                )
            ),
            step=1
        )


    with col2:

        days_to_followup = st.number_input(
            "Days to Follow-up",
            min_value=0,
            max_value=365,
            value=int(
                float(
                    sample.get(
                        "Days_to_Followup",
                        30
                    )
                )
            ),
            step=1
        )


    with col3:

        side_effect_options = [
            "None",
            "Mild",
            "Moderate",
            "Severe"
        ]

        side_default = sample.get(
            "Side_Effects",
            "None"
        )

        side_index = (
            side_effect_options.index(
                side_default
            )
            if side_default in side_effect_options
            else 0
        )

        side_effects = st.selectbox(
            "Side Effects",
            side_effect_options,
            index=side_index
        )


    adherence_default = int(
        float(
            sample.get(
                "Treatment_Adherence_Percent",
                75
            )
        )
    )

    adherence_default = max(
        25,
        min(
            100,
            adherence_default
        )
    )


    treatment_adherence = st.slider(
        "Treatment Adherence",
        min_value=25,
        max_value=100,
        value=adherence_default,
        step=1
    )


    # ========================================================
    # CREATE PATIENT DATA
    # ========================================================

    patient_data = {

        "Age": age,

        "Gender": gender,

        "BMI": bmi,

        "Smoking_Status":
            smoking_status,

        "Alcohol_Use":
            alcohol_use,

        "Physical_Activity":
            physical_activity,

        "Family_History":
            medical_values[
                "Family_History"
            ],

        "Diabetes":
            medical_values[
                "Diabetes"
            ],

        "Hypertension":
            medical_values[
                "Hypertension"
            ],

        "Heart_Disease":
            medical_values[
                "Heart_Disease"
            ],

        "Kidney_Disease":
            medical_values[
                "Kidney_Disease"
            ],

        "Respiratory_Disease":
            medical_values[
                "Respiratory_Disease"
            ],

        "Systolic_BP":
            systolic_bp,

        "Diastolic_BP":
            diastolic_bp,

        "Glucose_mg_dL":
            glucose,

        "Cholesterol_mg_dL":
            cholesterol,

        "Heart_Rate_bpm":
            heart_rate,

        "Disease_Category":
            disease_category,

        "Treatment_Plan":
            treatment_plan,

        "Treatment_Duration_Days":
            treatment_duration,

        "Treatment_Adherence_Percent":
            treatment_adherence,

        "Medication_Count":
            medication_count,

        "Previous_Hospitalizations":
            previous_hospitalizations,

        "Followup_Visits":
            followup_visits,

        "Days_to_Followup":
            days_to_followup,

        "Side_Effects":
            side_effects
    }


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("---")

    if st.button(
        "🔮 Predict Outcome",
        type="primary",
        use_container_width=True
    ):

        validation_errors = validate_patient(
            patient_data
        )

        if validation_errors:

            for error in validation_errors:

                st.error(error)

        else:

            try:

                with st.spinner(
                    "Analyzing patient data..."
                ):

                    prediction = make_prediction(
                        patient_data
                    )

                    save_patient_record(
                        patient_data,
                        prediction
                    )

                    st.session_state[
                        "prediction"
                    ] = prediction

                    st.session_state.pop(
                        "sample_patient",
                        None
                    )


                if prediction.lower() == "improved":

                    st.success(
                        f"AI Prediction: {prediction}"
                    )

                elif prediction.lower() == "stable":

                    st.info(
                        f"AI Prediction: {prediction}"
                    )

                elif prediction.lower() == "deteriorated":

                    st.error(
                        f"AI Prediction: {prediction}"
                    )

                else:

                    st.warning(
                        f"AI Prediction: {prediction}"
                    )


                st.success(
                    "Prediction saved successfully to the patient database."
                )


                # st.markdown(
                #     """
                #     <div class="info-box">
                #     The prediction was generated using the trained machine
                #     learning model and the result has been stored in the
                #     application database.
                #     </div>
                #     """,
                #     unsafe_allow_html=True
                # )


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.markdown(
        '<div class="main-title">Patient History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        View previously submitted patient information and their predicted health outcomes.
        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "↻ Refresh"
    ):

        get_patient_records.clear()

        st.rerun()


    patients = get_patient_records()


    if patients.empty:

        st.info(
            "No patient records found."
        )

    else:

        # --------------------------------------------
        # Filters
        # --------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            outcome_filter = st.selectbox(
                "Prediction",
                [
                    "All",
                    "Improved",
                    "Stable",
                    "Deteriorated"
                ]
            )


        with col2:

            diseases = [
                "All"
            ] + sorted(
                patients[
                    "Disease_Category"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            disease_filter = st.selectbox(
                "Disease",
                diseases
            )


        with col3:

            search = st.text_input(
                "Search Patient Records",
                placeholder="Search..."
            )


        filtered = patients.copy()


        if outcome_filter != "All":

            filtered = filtered[
                filtered["prediction"]
                .astype(str)
                .str.lower()
                ==
                outcome_filter.lower()
            ]


        if disease_filter != "All":

            filtered = filtered[
                filtered[
                    "Disease_Category"
                ].astype(str)
                ==
                disease_filter
            ]


        if search:

            mask = (
                filtered.astype(str)
                .apply(
                    lambda row:
                    row.str.contains(
                        search,
                        case=False,
                        na=False
                    ).any(),
                    axis=1
                )
            )

            filtered = filtered[
                mask
            ]


        # --------------------------------------------
        # Table matching React version
        # --------------------------------------------

        display_df = filtered[
            [
                "id",
                "Age",
                "Gender",
                "Disease_Category",
                "Treatment_Plan",
                "Treatment_Adherence_Percent",
                "prediction",
                "created_at"
            ]
        ].copy()


        display_df.columns = [
            "ID",
            "Age",
            "Gender",
            "Disease",
            "Treatment",
            "Adherence",
            "Prediction",
            "Date"
        ]


        display_df["ID"] = (
            "#"
            +
            display_df["ID"]
            .astype(str)
        )


        display_df[
            "Adherence"
        ] = (
            display_df[
                "Adherence"
            ].astype(str)
            +
            "%"
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        st.caption(
            f"Showing {len(filtered)} of {len(patients)} patient records."
        )


# ============================================================
# DISEASE ANALYSIS
# ============================================================

elif page == "🦠 Disease Analysis":

    st.markdown(
        '<div class="main-title">Disease Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Analyze patient records, predicted outcomes and treatment-related information for major health conditions.
        </div>
        """,
        unsafe_allow_html=True
    )

    patients = get_patient_records()

    if patients.empty:

        st.info(
            "No patient records available."
        )

    else:

        # ========================================================
        # DISEASE DEFINITIONS
        # ========================================================

        disease_definitions = {
            "Diabetes": {
                "column": "Diabetes",
                "icon": "🩸"
            },
            "Hypertension": {
                "column": "Hypertension",
                "icon": "🫀"
            },
            "Heart Disease": {
                "column": "Heart_Disease",
                "icon": "❤️"
            },
            "Kidney Disease": {
                "column": "Kidney_Disease",
                "icon": "🫘"
            },
            "Respiratory Disease": {
                "column": "Respiratory_Disease",
                "icon": "🫁"
            }
        }

        tabs = st.tabs(
            [
                "🩸 Diabetes",
                "🫀 Hypertension",
                "❤️ Heart Disease",
                "🫘 Kidney Disease",
                "🫁 Respiratory Disease"
            ]
        )

        # ========================================================
        # COMMON DISEASE TAB CONTENT
        # ========================================================

        def render_disease_analysis(tab, disease_name, disease_column, icon):

            with tab:

                st.markdown(
                    f'<div class="section-title">{icon} {disease_name} Analysis</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="section-description">Analysis of patients marked with {disease_name.lower()} in the stored patient records.</div>',
                    unsafe_allow_html=True
                )

                # --------------------------------------------
                # Filter patients with the selected disease
                # --------------------------------------------

                disease_patients = patients[
                    patients[disease_column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["yes", "1", "true"])
                ].copy()

                disease_count = len(disease_patients)
                total_count = len(patients)

                disease_percentage = (
                    (disease_count / total_count) * 100
                    if total_count > 0
                    else 0
                )

                # --------------------------------------------
                # Basic overview metrics
                # --------------------------------------------

                age = pd.to_numeric(
                    disease_patients["Age"],
                    errors="coerce"
                )

                bmi = pd.to_numeric(
                    disease_patients["BMI"],
                    errors="coerce"
                )

                adherence = pd.to_numeric(
                    disease_patients["Treatment_Adherence_Percent"],
                    errors="coerce"
                )

                hospitalizations = pd.to_numeric(
                    disease_patients["Previous_Hospitalizations"],
                    errors="coerce"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    f"{disease_name} Patients",
                    disease_count
                )

                c2.metric(
                    "% of Stored Patients",
                    f"{disease_percentage:.1f}%"
                )

                c3.metric(
                    "Average Age",
                    f"{age.mean():.1f}" if disease_count > 0 else "—"
                )

                c4.metric(
                    "Average BMI",
                    f"{bmi.mean():.1f}" if disease_count > 0 else "—"
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Average Treatment Adherence",
                    f"{adherence.mean():.1f}%" if disease_count > 0 else "—"
                )

                c2.metric(
                    "Average Hospitalizations",
                    f"{hospitalizations.mean():.1f}" if disease_count > 0 else "—"
                )

                if disease_count == 0:

                    st.info(
                        f"No patient records marked with {disease_name.lower()} are currently available."
                    )

                    return

                # --------------------------------------------
                # Predicted outcome analysis
                # --------------------------------------------

                st.markdown(
                    '<div class="section-title">Predicted Outcome Analysis</div>',
                    unsafe_allow_html=True
                )

                outcome_series = (
                    disease_patients["prediction"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                )

                outcome_df = pd.DataFrame(
                    {
                        "Outcome": [
                            "Improved",
                            "Stable",
                            "Deteriorated"
                        ],
                        "Patients": [
                            outcome_series.eq("Improved").sum(),
                            outcome_series.eq("Stable").sum(),
                            outcome_series.eq("Deteriorated").sum()
                        ]
                    }
                )

                outcome_col1, outcome_col2 = st.columns(2)

                with outcome_col1:

                    fig = px.bar(
                        outcome_df,
                        x="Outcome",
                        y="Patients",
                        title=f"Predicted Outcomes Among {disease_name} Patients"
                    )

                    fig.update_layout(
                        showlegend=False,
                        height=350
                    )

                    st.plotly_chart(
                        style_plotly(fig),
                        use_container_width=True
                    )

                with outcome_col2:

                    fig = px.pie(
                        outcome_df,
                        names="Outcome",
                        values="Patients",
                        title=f"Outcome Percentage — {disease_name}"
                    )

                    fig.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        style_plotly(fig),
                        use_container_width=True
                    )

                st.info(
                    f"Among patients marked with {disease_name.lower()} in the stored records, "
                    f"{outcome_series.eq('Deteriorated').sum()} of {disease_count} "
                    f"({(outcome_series.eq('Deteriorated').sum() / disease_count) * 100:.1f}%) "
                    "were predicted as Deteriorated by the machine learning model."
                )

                # --------------------------------------------
                # Treatment and follow-up insights
                # --------------------------------------------

                st.markdown(
                    '<div class="section-title">Additional Insights</div>',
                    unsafe_allow_html=True
                )

                insight_col1, insight_col2 = st.columns(2)

                with insight_col1:

                    treatment_counts = (
                        disease_patients["Treatment_Plan"]
                        .fillna("Unknown")
                        .astype(str)
                        .value_counts()
                        .reset_index()
                    )

                    treatment_counts.columns = [
                        "Treatment",
                        "Patients"
                    ]

                    fig = px.bar(
                        treatment_counts,
                        x="Treatment",
                        y="Patients",
                        title=f"Treatment Plans — {disease_name}"
                    )

                    fig.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        style_plotly(fig),
                        use_container_width=True
                    )

                with insight_col2:

                    hospitalization_counts = (
                        disease_patients["Previous_Hospitalizations"]
                        .value_counts()
                        .sort_index()
                        .reset_index()
                    )

                    hospitalization_counts.columns = [
                        "Hospitalizations",
                        "Patients"
                    ]

                    fig = px.bar(
                        hospitalization_counts,
                        x="Hospitalizations",
                        y="Patients",
                        title=f"Previous Hospitalizations — {disease_name}"
                    )

                    fig.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        style_plotly(fig),
                        use_container_width=True
                    )

                # --------------------------------------------
                # Follow-up and adherence statistics
                # --------------------------------------------

                followup_visits = pd.to_numeric(
                    disease_patients["Followup_Visits"],
                    errors="coerce"
                )

                days_to_followup = pd.to_numeric(
                    disease_patients["Days_to_Followup"],
                    errors="coerce"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Average Follow-up Visits",
                    f"{followup_visits.mean():.1f}"
                )

                c2.metric(
                    "Average Days to Follow-up",
                    f"{days_to_followup.mean():.1f}"
                )

                c3.metric(
                    "Average Adherence",
                    f"{adherence.mean():.1f}%"
                )

                # --------------------------------------------
                # Patient records
                # --------------------------------------------

                st.markdown(
                    f'<div class="section-title">{disease_name} Patient Records</div>',
                    unsafe_allow_html=True
                )

                display = disease_patients[
                    [
                        "id",
                        "Age",
                        "Gender",
                        "Disease_Category",
                        "Treatment_Plan",
                        "Treatment_Adherence_Percent",
                        "prediction",
                        "created_at"
                    ]
                ].copy()

                display.columns = [
                    "ID",
                    "Age",
                    "Gender",
                    "Disease",
                    "Treatment",
                    "Adherence",
                    "Prediction",
                    "Date"
                ]

                display["Adherence"] = (
                    display["Adherence"].astype(str) + "%"
                )

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    f"Showing {len(disease_patients)} patient records marked with {disease_name.lower()}."
                )

        for tab, (disease_name, details) in zip(
            tabs,
            disease_definitions.items()
        ):

            render_disease_analysis(
                tab,
                disease_name,
                details["column"],
                details["icon"]
            )


# ============================================================
# TREATMENT ANALYSIS
# ============================================================

elif page == "💊 Treatment Analysis":

    st.markdown(
        '<div class="main-title">Treatment Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Analyze treatment plans, adherence and predicted outcomes.
        </div>
        """,
        unsafe_allow_html=True
    )


    patients = get_patient_records()


    if patients.empty:

        st.info(
            "No patient records available."
        )

    else:

        # ----------------------------------------
        # Treatment distribution
        # ----------------------------------------

        treatment_counts = (
            patients[
                "Treatment_Plan"
            ]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )


        treatment_counts.columns = [
            "Treatment",
            "Patients"
        ]


        fig = px.bar(
            treatment_counts,
            x="Treatment",
            y="Patients",
            title="Treatment Plan Distribution"
        )


        st.plotly_chart(
            style_plotly(fig),
            use_container_width=True
        )


        # ----------------------------------------
        # Treatment vs outcome
        # ----------------------------------------

        treatment_outcomes = pd.crosstab(
            patients[
                "Treatment_Plan"
            ],
            patients[
                "prediction"
            ]
        ).reset_index()


        melted = treatment_outcomes.melt(
            id_vars=[
                "Treatment_Plan"
            ],
            var_name="Outcome",
            value_name="Patients"
        )


        fig = px.bar(
            melted,
            x="Treatment_Plan",
            y="Patients",
            color="Outcome",
            barmode="group",
            title="Treatment Plan vs Predicted Outcome"
        )


        st.plotly_chart(
            style_plotly(fig),
            use_container_width=True
        )


        # ----------------------------------------
        # Adherence
        # ----------------------------------------

        adherence = pd.to_numeric(
            patients[
                "Treatment_Adherence_Percent"
            ],
            errors="coerce"
        )


        fig = px.histogram(
            adherence,
            nbins=10,
            title="Treatment Adherence Distribution"
        )


        fig.update_layout(
            xaxis_title="Treatment Adherence (%)",
            yaxis_title="Number of Patients"
        )


        st.plotly_chart(
            style_plotly(fig),
            use_container_width=True
        )


        # ----------------------------------------
        # Treatment statistics
        # ----------------------------------------

        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Average Adherence",
            f"{adherence.mean():.1f}%"
        )


        duration = pd.to_numeric(
            patients[
                "Treatment_Duration_Days"
            ],
            errors="coerce"
        )


        c2.metric(
            "Average Treatment Duration",
            f"{duration.mean():.1f} days"
        )


        medication = pd.to_numeric(
            patients[
                "Medication_Count"
            ],
            errors="coerce"
        )


        c3.metric(
            "Average Medication Count",
            f"{medication.mean():.1f}"
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "⚠️ Risk Analysis":

    st.markdown(
        '<div class="main-title">Risk Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Analysis of records predicted as Deteriorated by the machine learning model.
        </div>
        """,
        unsafe_allow_html=True
    )


    patients = get_patient_records()


    if patients.empty:

        st.info(
            "No patient records available."
        )

    else:

        risk_patients = patients[
            patients[
                "prediction"
            ]
            .astype(str)
            .str.lower()
            ==
            "deteriorated"
        ].copy()


        risk_count = len(
            risk_patients
        )


        st.metric(
            "Deteriorated Predictions",
            risk_count
        )


        if risk_count == 0:

            st.success(
                "No records are currently predicted as Deteriorated."
            )

        else:

            age = pd.to_numeric(
                risk_patients[
                    "Age"
                ],
                errors="coerce"
            )

            bmi = pd.to_numeric(
                risk_patients[
                    "BMI"
                ],
                errors="coerce"
            )

            adherence = pd.to_numeric(
                risk_patients[
                    "Treatment_Adherence_Percent"
                ],
                errors="coerce"
            )

            hospitalizations = pd.to_numeric(
                risk_patients[
                    "Previous_Hospitalizations"
                ],
                errors="coerce"
            )


            c1, c2, c3, c4 = st.columns(4)


            c1.metric(
                "Average Age",
                f"{age.mean():.1f}"
            )

            c2.metric(
                "Average BMI",
                f"{bmi.mean():.1f}"
            )

            c3.metric(
                "Average Adherence",
                f"{adherence.mean():.1f}%"
            )

            c4.metric(
                "Avg. Hospitalizations",
                f"{hospitalizations.mean():.1f}"
            )


            # ----------------------------------------
            # Disease distribution
            # ----------------------------------------

            risk_diseases = (
                risk_patients[
                    "Disease_Category"
                ]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )


            risk_diseases.columns = [
                "Disease",
                "Patients"
            ]


            fig = px.bar(
                risk_diseases,
                x="Disease",
                y="Patients",
                title="Disease Distribution Among Deteriorated Predictions"
            )


            st.plotly_chart(
                style_plotly(fig),
                use_container_width=True
            )


            # ----------------------------------------
            # Records
            # ----------------------------------------

            st.markdown(
                "### Deteriorated Patient Records"
            )


            st.dataframe(
                risk_patients,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

# st.markdown(
#     """
#     <div class="footer">
#         Patient Health Outcome Predictor • Machine Learning Based Decision Support System
#     </div>
#     """,
#     unsafe_allow_html=True
# )










# import streamlit as st
# import pandas as pd
# import numpy as np
# import sqlite3
# import joblib
# from pathlib import Path
# import plotly.express as px


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Patient Health Outcome Predictor",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # ============================================================
# # PROJECT PATHS
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent

# DATABASE_PATH = BASE_DIR / "backend" / "database.db"

# MODEL_PATH = (
#     BASE_DIR
#     / "backend"
#     / "model"
#     / "logistic_regression_model.pkl"
# )

# PREPROCESSOR_PATH = (
#     BASE_DIR
#     / "backend"
#     / "model"
#     / "preprocessor.pkl"
# )

# DATASET_PATH = (
#     BASE_DIR
#     / "data"
#     / "patient_health_outcome_predictor_30000.csv"
# )


# # ============================================================
# # CUSTOM CSS
# # ============================================================

# st.markdown(
#     """
#     <style>

#     /* ---------- Main Page ---------- */

#     .main {
#         background-color: #f7f8fa;
#     }

#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 3rem;
#     }


#     /* ---------- Sidebar ---------- */

#     section[data-testid="stSidebar"] {
#         background-color: #111827;
#     }

#     section[data-testid="stSidebar"] * {
#         color: white;
#     }


#     /* ---------- Header ---------- */

#     .main-title {
#         font-size: 32px;
#         font-weight: 700;
#         margin-bottom: 4px;
#         color: #FFFF00;
#     }

#     .subtitle {
#         color: #6b7280;
#         font-size: 15px;
#         margin-bottom: 25px;
#     }


#     /* ---------- Cards ---------- */

#     .metric-card {
#         background: white;
#         padding: 20px;
#         border-radius: 12px;
#         border: 1px solid #e5e7eb;
#         min-height: 120px;
#     }

#     .metric-title {
#         color: #6b7280;
#         font-size: 14px;
#         margin-bottom: 8px;
#     }

#     .metric-value {
#         color: #FFFF00;
#         font-size: 28px;
#         font-weight: 700;
#     }


#     /* ---------- Prediction Result ---------- */

#     .prediction-result {
#         padding: 25px;
#         border-radius: 12px;
#         text-align: center;
#         margin-top: 20px;
#         margin-bottom: 20px;
#     }

#     .prediction-title {
#         font-size: 14px;
#         font-weight: 600;
#         letter-spacing: 1px;
#         margin-bottom: 8px;
#     }

#     .prediction-value {
#         font-size: 32px;
#         font-weight: 700;
#     }


#     /* ---------- Section ---------- */

#     .section-title {
#         font-size: 21px;
#         font-weight: 700;
#         color: #FFFF00;
#         margin-top: 25px;
#         margin-bottom: 5px;
#     }

#     .section-description {
#         color: #6b7280;
#         margin-bottom: 18px;
#     }


#     /* ---------- Info ---------- */

#     .info-box {
#         padding: 16px;
#         border-radius: 10px;
#         background-color: #eff6ff;
#         border: 1px solid #bfdbfe;
#         margin-top: 10px;
#         margin-bottom: 15px;
#     }


#     /* ---------- Footer ---------- */

#     .footer {
#         text-align: center;
#         color: #9ca3af;
#         font-size: 13px;
#         padding-top: 40px;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # ============================================================
# # LOAD MODEL
# # ============================================================

# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)


# @st.cache_resource
# def load_preprocessor():
#     return joblib.load(PREPROCESSOR_PATH)


# # ============================================================
# # LOAD TRAINING DATASET
# # ============================================================

# @st.cache_data
# def load_dataset():
#     return pd.read_csv(DATASET_PATH)


# # ============================================================
# # DATABASE FUNCTIONS
# # ============================================================

# def get_connection():
#     return sqlite3.connect(DATABASE_PATH)


# @st.cache_data(ttl=2)
# def get_patient_records():
#     """
#     Read patient records from SQLite database.
#     """

#     try:

#         conn = get_connection()

#         query = """
#         SELECT *
#         FROM patient_records
#         ORDER BY id ASC
#         """

#         df = pd.read_sql_query(query, conn)

#         conn.close()

#         return df

#     except Exception as e:

#         st.error(f"Unable to load patient records: {e}")

#         return pd.DataFrame()


# def save_patient_record(patient_data, prediction):

#     conn = get_connection()

#     cursor = conn.cursor()

#     query = """
#     INSERT INTO patient_records (
#         Age,
#         Gender,
#         BMI,
#         Smoking_Status,
#         Alcohol_Use,
#         Physical_Activity,
#         Family_History,
#         Diabetes,
#         Hypertension,
#         Heart_Disease,
#         Kidney_Disease,
#         Respiratory_Disease,
#         Systolic_BP,
#         Diastolic_BP,
#         Glucose_mg_dL,
#         Cholesterol_mg_dL,
#         Heart_Rate_bpm,
#         Disease_Category,
#         Treatment_Plan,
#         Treatment_Duration_Days,
#         Treatment_Adherence_Percent,
#         Medication_Count,
#         Previous_Hospitalizations,
#         Followup_Visits,
#         Days_to_Followup,
#         Side_Effects,
#         prediction,
#         created_at
#     )
#     VALUES (
#         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
#         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime')
#     )
#     """

#     values = (
#         patient_data["Age"],
#         patient_data["Gender"],
#         patient_data["BMI"],
#         patient_data["Smoking_Status"],
#         patient_data["Alcohol_Use"],
#         patient_data["Physical_Activity"],
#         patient_data["Family_History"],
#         patient_data["Diabetes"],
#         patient_data["Hypertension"],
#         patient_data["Heart_Disease"],
#         patient_data["Kidney_Disease"],
#         patient_data["Respiratory_Disease"],
#         patient_data["Systolic_BP"],
#         patient_data["Diastolic_BP"],
#         patient_data["Glucose_mg_dL"],
#         patient_data["Cholesterol_mg_dL"],
#         patient_data["Heart_Rate_bpm"],
#         patient_data["Disease_Category"],
#         patient_data["Treatment_Plan"],
#         patient_data["Treatment_Duration_Days"],
#         patient_data["Treatment_Adherence_Percent"],
#         patient_data["Medication_Count"],
#         patient_data["Previous_Hospitalizations"],
#         patient_data["Followup_Visits"],
#         patient_data["Days_to_Followup"],
#         patient_data["Side_Effects"],
#         prediction
#     )

#     cursor.execute(query, values)

#     conn.commit()

#     conn.close()

#     get_patient_records.clear()


# # ============================================================
# # DATASET OPTIONS
# # ============================================================

# def get_options(column):

#     dataset = load_dataset()

#     if column not in dataset.columns:
#         return []

#     values = (
#         dataset[column]
#         .dropna()
#         .astype(str)
#         .unique()
#         .tolist()
#     )

#     return sorted(values)


# # ============================================================
# # RANDOM SAMPLE
# # ============================================================

# def generate_random_sample():

#     dataset = load_dataset()

#     sample = dataset.sample(
#         n=1
#     ).iloc[0]

#     patient = {}

#     feature_columns = [
#         "Age",
#         "Gender",
#         "BMI",
#         "Smoking_Status",
#         "Alcohol_Use",
#         "Physical_Activity",
#         "Family_History",
#         "Diabetes",
#         "Hypertension",
#         "Heart_Disease",
#         "Kidney_Disease",
#         "Respiratory_Disease",
#         "Systolic_BP",
#         "Diastolic_BP",
#         "Glucose_mg_dL",
#         "Cholesterol_mg_dL",
#         "Heart_Rate_bpm",
#         "Disease_Category",
#         "Treatment_Plan",
#         "Treatment_Duration_Days",
#         "Treatment_Adherence_Percent",
#         "Medication_Count",
#         "Previous_Hospitalizations",
#         "Followup_Visits",
#         "Days_to_Followup",
#         "Side_Effects"
#     ]

#     for column in feature_columns:

#         if column in dataset.columns:

#             value = sample[column]

#             if pd.isna(value):
#                 value = ""

#             patient[column] = value

#     return patient


# # ============================================================
# # MODEL PREDICTION
# # ============================================================

# def make_prediction(patient_data):

#     model = load_model()

#     preprocessor = load_preprocessor()

#     input_df = pd.DataFrame(
#         [patient_data]
#     )

#     transformed_data = preprocessor.transform(
#         input_df
#     )

#     prediction = model.predict(
#         transformed_data
#     )[0]

#     return str(prediction)


# # ============================================================
# # VALIDATION
# # ============================================================

# def validate_patient(data):

#     errors = []

#     # ----------------------------
#     # Required fields
#     # ----------------------------

#     required_fields = [
#         "Age",
#         "BMI",
#         "Gender",
#         "Smoking_Status",
#         "Alcohol_Use",
#         "Physical_Activity",
#         "Family_History",
#         "Diabetes",
#         "Hypertension",
#         "Heart_Disease",
#         "Kidney_Disease",
#         "Respiratory_Disease",
#         "Systolic_BP",
#         "Diastolic_BP",
#         "Glucose_mg_dL",
#         "Cholesterol_mg_dL",
#         "Heart_Rate_bpm",
#         "Disease_Category",
#         "Treatment_Plan",
#         "Treatment_Duration_Days",
#         "Treatment_Adherence_Percent",
#         "Medication_Count",
#         "Previous_Hospitalizations",
#         "Followup_Visits",
#         "Days_to_Followup",
#         "Side_Effects"
#     ]

#     for field in required_fields:

#         value = data.get(field)

#         if value is None or value == "":

#             errors.append(
#                 f"Please provide {field.replace('_', ' ')}."
#             )


#     # ----------------------------
#     # Numeric validation
#     # ----------------------------

#     try:

#         age = float(data["Age"])

#         if age < 18 or age > 90:

#             errors.append(
#                 "Age must be between 18 and 90."
#             )

#     except:
#         pass


#     try:

#         bmi = float(data["BMI"])

#         if bmi < 15 or bmi > 45:

#             errors.append(
#                 "BMI must be between 15 and 45."
#             )

#     except:
#         pass


#     try:

#         systolic = float(
#             data["Systolic_BP"]
#         )

#         if systolic < 80 or systolic > 250:

#             errors.append(
#                 "Systolic BP must be between 80 and 250 mmHg."
#             )

#     except:
#         pass


#     try:

#         diastolic = float(
#             data["Diastolic_BP"]
#         )

#         if diastolic < 40 or diastolic > 150:

#             errors.append(
#                 "Diastolic BP must be between 40 and 150 mmHg."
#             )

#     except:
#         pass


#     try:

#         systolic = float(
#             data["Systolic_BP"]
#         )

#         diastolic = float(
#             data["Diastolic_BP"]
#         )

#         if systolic <= diastolic:

#             errors.append(
#                 "Systolic BP should be greater than diastolic BP."
#             )

#     except:
#         pass


#     try:

#         glucose = float(
#             data["Glucose_mg_dL"]
#         )

#         if glucose < 50 or glucose > 500:

#             errors.append(
#                 "Glucose must be between 50 and 500 mg/dL."
#             )

#     except:
#         pass


#     try:

#         cholesterol = float(
#             data["Cholesterol_mg_dL"]
#         )

#         if cholesterol < 100 or cholesterol > 500:

#             errors.append(
#                 "Cholesterol must be between 100 and 500 mg/dL."
#             )

#     except:
#         pass


#     try:

#         heart_rate = float(
#             data["Heart_Rate_bpm"]
#         )

#         if heart_rate < 30 or heart_rate > 220:

#             errors.append(
#                 "Heart rate must be between 30 and 220 bpm."
#             )

#     except:
#         pass


#     try:

#         duration = float(
#             data["Treatment_Duration_Days"]
#         )

#         if duration < 1 or duration > 1000:

#             errors.append(
#                 "Treatment duration must be between 1 and 1000 days."
#             )

#     except:
#         pass


#     try:

#         adherence = float(
#             data["Treatment_Adherence_Percent"]
#         )

#         if adherence < 25 or adherence > 100:

#             errors.append(
#                 "Treatment adherence must be between 25% and 100%."
#             )

#     except:
#         pass


#     try:

#         medications = float(
#             data["Medication_Count"]
#         )

#         if medications < 0 or medications > 10:

#             errors.append(
#                 "Medication count must be between 0 and 10."
#             )

#     except:
#         pass


#     try:

#         hospitalizations = float(
#             data["Previous_Hospitalizations"]
#         )

#         if hospitalizations < 0 or hospitalizations > 6:

#             errors.append(
#                 "Previous hospitalizations must be between 0 and 6."
#             )

#     except:
#         pass


#     try:

#         followups = float(
#             data["Followup_Visits"]
#         )

#         if followups < 0 or followups > 7:

#             errors.append(
#                 "Follow-up visits must be between 0 and 7."
#             )

#     except:
#         pass


#     try:

#         days = float(
#             data["Days_to_Followup"]
#         )

#         if days < 0 or days > 365:

#             errors.append(
#                 "Days to follow-up must be between 0 and 365."
#             )

#     except:
#         pass


#     return errors


# # ============================================================
# # SIDEBAR
# # ============================================================

# with st.sidebar:

#     st.markdown(
#         """
#         <div style="
#             font-size:24px;
#             font-weight:700;
#             margin-bottom:5px;
#         ">
#             🏥 Patient Health
#         </div>

#         <div style="
#             font-size:13px;
#             color:#9ca3af;
#             margin-bottom:25px;
#         ">
#             Outcome Predictor
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     page = st.radio(
#         "Navigation",
#         [
#             "🏠 Dashboard",
#             "➕ New Prediction",
#             "📋 Patient History",
#             "❤️ Heart Disease",
#             "🦠 Disease Analysis",
#             "💊 Treatment Analysis",
#             "⚠️ Risk Analysis"
#         ]
#     )

#     st.markdown("---")

#     st.caption(
#         "AI-powered patient outcome prediction"
#     )


# # ============================================================
# # LOAD RECORDS
# # ============================================================

# patients = get_patient_records()


# # ============================================================
# # DASHBOARD
# # ============================================================

# if page == "🏠 Dashboard":

#     st.markdown(
#         '<div class="main-title">Dashboard</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         Overview of patient predictions and health outcome statistics.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     if st.button(
#         "➕ New Prediction"
#     ):

#         st.session_state["page"] = "➕ New Prediction"

#         st.rerun()


#     # ----------------------------------------
#     # Basic statistics
#     # ----------------------------------------

#     total_patients = len(patients)

#     improved = 0
#     stable = 0
#     deteriorated = 0

#     if total_patients > 0:

#         improved = (
#             patients["prediction"]
#             .astype(str)
#             .str.lower()
#             .eq("improved")
#             .sum()
#         )

#         stable = (
#             patients["prediction"]
#             .astype(str)
#             .str.lower()
#             .eq("stable")
#             .sum()
#         )

#         deteriorated = (
#             patients["prediction"]
#             .astype(str)
#             .str.lower()
#             .eq("deteriorated")
#             .sum()
#         )


#     # ----------------------------------------
#     # Metric cards
#     # ----------------------------------------

#     col1, col2, col3, col4 = st.columns(4)

#     col1.metric(
#         "Total Patients",
#         total_patients
#     )

#     col2.metric(
#         "Improved",
#         improved
#     )

#     col3.metric(
#         "Stable",
#         stable
#     )

#     col4.metric(
#         "Deteriorated",
#         deteriorated
#     )


#     st.markdown(
#         '<div class="section-title">Patient Statistics</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         '<div class="section-description">Average values from submitted patient records.</div>',
#         unsafe_allow_html=True
#     )


#     # ----------------------------------------
#     # Average statistics
#     # ----------------------------------------

#     if total_patients > 0:

#         average_age = pd.to_numeric(
#             patients["Age"],
#             errors="coerce"
#         ).mean()

#         average_bmi = pd.to_numeric(
#             patients["BMI"],
#             errors="coerce"
#         ).mean()

#         average_adherence = pd.to_numeric(
#             patients["Treatment_Adherence_Percent"],
#             errors="coerce"
#         ).mean()

#         average_hospitalizations = pd.to_numeric(
#             patients["Previous_Hospitalizations"],
#             errors="coerce"
#         ).mean()

#     else:

#         average_age = 0
#         average_bmi = 0
#         average_adherence = 0
#         average_hospitalizations = 0


#     c1, c2, c3, c4 = st.columns(4)

#     c1.metric(
#         "Average Age",
#         f"{average_age:.1f} years"
#     )

#     c2.metric(
#         "Average BMI",
#         f"{average_bmi:.1f}"
#     )

#     c3.metric(
#         "Average Treatment Adherence",
#         f"{average_adherence:.1f}%"
#     )

#     c4.metric(
#         "Avg. Hospitalizations",
#         f"{average_hospitalizations:.1f}"
#     )


#     # ----------------------------------------
#     # Outcome Distribution
#     # ----------------------------------------

#     st.markdown(
#         '<div class="section-title">Outcome Distribution</div>',
#         unsafe_allow_html=True
#     )

#     if total_patients == 0:

#         st.info(
#             "No prediction data available."
#         )

#     else:

#         outcome_df = pd.DataFrame(
#             {
#                 "Outcome": [
#                     "Improved",
#                     "Stable",
#                     "Deteriorated"
#                 ],
#                 "Patients": [
#                     improved,
#                     stable,
#                     deteriorated
#                 ]
#             }
#         )

#         col1, col2 = st.columns(2)

#         with col1:

#             fig = px.bar(
#                 outcome_df,
#                 x="Outcome",
#                 y="Patients",
#                 title="Predicted Patient Outcomes"
#             )

#             fig.update_layout(
#                 showlegend=False,
#                 height=350
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )

#         with col2:

#             fig = px.pie(
#                 outcome_df,
#                 names="Outcome",
#                 values="Patients",
#                 title="Outcome Percentage"
#             )

#             fig.update_layout(
#                 height=350
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )


#     # ----------------------------------------
#     # Recent predictions
#     # ----------------------------------------

#     st.markdown(
#         '<div class="section-title">Recent Predictions</div>',
#         unsafe_allow_html=True
#     )

#     if total_patients == 0:

#         st.info(
#             "No patient records available."
#         )

#     else:

#         recent_patients = (
#             patients
#             .sort_values(
#                 "id",
#                 ascending=False
#             )
#             .head(5)
#             .sort_values(
#                 "id",
#                 ascending=True
#             )
#         )

#         recent_display = recent_patients[
#             [
#                 "id",
#                 "Age",
#                 "Disease_Category",
#                 "prediction",
#                 "created_at"
#             ]
#         ].copy()

#         recent_display.columns = [
#             "Patient ID",
#             "Age",
#             "Disease",
#             "Prediction",
#             "Date"
#         ]

#         st.dataframe(
#             recent_display,
#             use_container_width=True,
#             hide_index=True
#         )


#     # ----------------------------------------
#     # Disease distribution
#     # ----------------------------------------

#     st.markdown(
#         '<div class="section-title">Disease Distribution</div>',
#         unsafe_allow_html=True
#     )

#     if total_patients > 0:

#         disease_counts = (
#             patients["Disease_Category"]
#             .fillna("Unknown")
#             .value_counts()
#             .reset_index()
#         )

#         disease_counts.columns = [
#             "Disease",
#             "Patients"
#         ]

#         col1, col2 = st.columns(2)

#         with col1:

#             fig = px.bar(
#                 disease_counts,
#                 x="Disease",
#                 y="Patients",
#                 title="Patients by Disease Category"
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )

#         with col2:

#             fig = px.pie(
#                 disease_counts,
#                 names="Disease",
#                 values="Patients",
#                 title="Disease Percentage"
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )

#     else:

#         st.info(
#             "No disease data available."
#         )


#     # ----------------------------------------
#     # Heart Disease overview
#     # ----------------------------------------

#     st.markdown(
#         '<div class="section-title">Heart Disease Overview</div>',
#         unsafe_allow_html=True
#     )

#     if total_patients > 0:

#         heart_patients = patients[
#             patients["Heart_Disease"]
#             .astype(str)
#             .str.strip()
#             .str.lower()
#             .isin(
#                 ["yes", "1", "true"]
#             )
#         ]

#         heart_count = len(
#             heart_patients
#         )

#         st.metric(
#             "Patients with Heart Disease",
#             heart_count
#         )

#         if heart_count > 0:

#             heart_outcomes = (
#                 heart_patients["prediction"]
#                 .value_counts()
#                 .reset_index()
#             )

#             heart_outcomes.columns = [
#                 "Outcome",
#                 "Patients"
#             ]

#             fig = px.bar(
#                 heart_outcomes,
#                 x="Outcome",
#                 y="Patients",
#                 title="Predicted Outcomes Among Heart Disease Patients"
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )


# # ============================================================
# # NEW PREDICTION
# # ============================================================

# elif page == "➕ New Prediction":

#     st.markdown(
#         '<div class="main-title">New Patient Prediction</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         Enter the patient's information below.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     # ========================================================
#     # SAMPLE / CLEAR BUTTONS
#     # ========================================================

#     col1, col2 = st.columns(
#         [1, 1]
#     )

#     with col1:

#         if st.button(
#             "🎲 Load Random Sample",
#             use_container_width=True
#         ):

#             st.session_state["sample_patient"] = (
#                 generate_random_sample()
#             )

#             st.session_state["prediction"] = None

#             st.rerun()


#     with col2:

#         if st.button(
#             "🗑️ Clear Form",
#             use_container_width=True
#         ):

#             st.session_state.pop(
#                 "sample_patient",
#                 None
#             )

#             st.session_state.pop(
#                 "prediction",
#                 None
#             )

#             st.rerun()


#     sample = st.session_state.get(
#         "sample_patient",
#         {}
#     )


#     # ========================================================
#     # SECTION 01 — DEMOGRAPHICS
#     # ========================================================

#     st.markdown(
#         '<div class="section-title">01 — Demographics</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         '<div class="section-description">Basic information about the patient.</div>',
#         unsafe_allow_html=True
#     )


#     col1, col2, col3 = st.columns(3)

#     with col1:

#         age = st.number_input(
#             "Age",
#             min_value=18,
#             max_value=90,
#             value=int(
#                 float(sample.get("Age", 45))
#             ),
#             step=1
#         )


#     with col2:

#         gender_options = [
#             "Male",
#             "Female",
#             "Other"
#         ]

#         gender_default = sample.get(
#             "Gender",
#             "Male"
#         )

#         gender_index = (
#             gender_options.index(
#                 gender_default
#             )
#             if gender_default in gender_options
#             else 0
#         )

#         gender = st.selectbox(
#             "Gender",
#             gender_options,
#             index=gender_index
#         )


#     with col3:

#         bmi = st.number_input(
#             "BMI",
#             min_value=15.0,
#             max_value=45.0,
#             value=float(
#                 sample.get("BMI", 25.0)
#             ),
#             step=0.1
#         )


#     col1, col2, col3 = st.columns(3)

#     with col1:

#         smoking_options = [
#             "Never",
#             "Former",
#             "Current"
#         ]

#         smoking_default = sample.get(
#             "Smoking_Status",
#             "Never"
#         )

#         smoking_index = (
#             smoking_options.index(
#                 smoking_default
#             )
#             if smoking_default in smoking_options
#             else 0
#         )

#         smoking_status = st.selectbox(
#             "Smoking Status",
#             smoking_options,
#             index=smoking_index
#         )


#     with col2:

#         alcohol_options = [
#             "Occasional",
#             "Regular",
#             "Unknown"
#         ]

#         alcohol_default = sample.get(
#             "Alcohol_Use",
#             "Unknown"
#         )

#         alcohol_index = (
#             alcohol_options.index(
#                 alcohol_default
#             )
#             if alcohol_default in alcohol_options
#             else 0
#         )

#         alcohol_use = st.selectbox(
#             "Alcohol Use",
#             alcohol_options,
#             index=alcohol_index
#         )


#     with col3:

#         activity_options = [
#             "Low",
#             "Moderate",
#             "High"
#         ]

#         activity_default = sample.get(
#             "Physical_Activity",
#             "Moderate"
#         )

#         activity_index = (
#             activity_options.index(
#                 activity_default
#             )
#             if activity_default in activity_options
#             else 1
#         )

#         physical_activity = st.selectbox(
#             "Physical Activity",
#             activity_options,
#             index=activity_index
#         )


#     # ========================================================
#     # SECTION 02 — MEDICAL HISTORY
#     # ========================================================

#     st.markdown(
#         '<div class="section-title">02 — Medical History</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         '<div class="section-description">Existing conditions and medical background.</div>',
#         unsafe_allow_html=True
#     )


#     medical_fields = [
#         ("Family_History", "Family History"),
#         ("Diabetes", "Diabetes"),
#         ("Hypertension", "Hypertension"),
#         ("Heart_Disease", "Heart Disease"),
#         ("Kidney_Disease", "Kidney Disease"),
#         ("Respiratory_Disease", "Respiratory Disease")
#     ]


#     medical_values = {}


#     cols = st.columns(3)

#     for i, (field, label) in enumerate(
#         medical_fields
#     ):

#         with cols[i % 3]:

#             default_value = sample.get(
#                 field,
#                 "No"
#             )

#             options = [
#                 "Yes",
#                 "No"
#             ]

#             index = (
#                 options.index(
#                     default_value
#                 )
#                 if default_value in options
#                 else 1
#             )

#             medical_values[field] = st.selectbox(
#                 label,
#                 options,
#                 index=index,
#                 key=f"medical_{field}"
#             )


#     # ========================================================
#     # SECTION 03 — CLINICAL MEASUREMENTS
#     # ========================================================

#     st.markdown(
#         '<div class="section-title">03 — Clinical Measurements</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         '<div class="section-description">Current clinical and vital measurements.</div>',
#         unsafe_allow_html=True
#     )


#     col1, col2 = st.columns(2)

#     with col1:

#         systolic_bp = st.number_input(
#             "Systolic BP (mmHg)",
#             min_value=80,
#             max_value=250,
#             value=int(
#                 float(
#                     sample.get(
#                         "Systolic_BP",
#                         120
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col2:

#         diastolic_bp = st.number_input(
#             "Diastolic BP (mmHg)",
#             min_value=40,
#             max_value=150,
#             value=int(
#                 float(
#                     sample.get(
#                         "Diastolic_BP",
#                         80
#                     )
#                 )
#             ),
#             step=1
#         )


#     col1, col2, col3 = st.columns(3)

#     with col1:

#         glucose = st.number_input(
#             "Glucose (mg/dL)",
#             min_value=50,
#             max_value=500,
#             value=int(
#                 float(
#                     sample.get(
#                         "Glucose_mg_dL",
#                         120
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col2:

#         cholesterol = st.number_input(
#             "Cholesterol (mg/dL)",
#             min_value=100,
#             max_value=500,
#             value=int(
#                 float(
#                     sample.get(
#                         "Cholesterol_mg_dL",
#                         200
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col3:

#         heart_rate = st.number_input(
#             "Heart Rate (bpm)",
#             min_value=30,
#             max_value=220,
#             value=int(
#                 float(
#                     sample.get(
#                         "Heart_Rate_bpm",
#                         75
#                     )
#                 )
#             ),
#             step=1
#         )


#     disease_options = [
#         "Diabetes",
#         "Cardiovascular",
#         "Respiratory",
#         "Kidney",
#         "Other"
#     ]

#     disease_default = sample.get(
#         "Disease_Category",
#         "Other"
#     )

#     disease_index = (
#         disease_options.index(
#             disease_default
#         )
#         if disease_default in disease_options
#         else 0
#     )

#     disease_category = st.selectbox(
#         "Disease Category",
#         disease_options,
#         index=disease_index
#     )


#     # ========================================================
#     # SECTION 04 — TREATMENT
#     # ========================================================

#     st.markdown(
#         '<div class="section-title">04 — Treatment Details</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         '<div class="section-description">Current treatment and patient adherence.</div>',
#         unsafe_allow_html=True
#     )


#     treatment_options = [
#         "Medication",
#         "Surgery",
#         "Therapy",
#         "Lifestyle",
#         "Medication + Therapy"
#     ]

#     treatment_default = sample.get(
#         "Treatment_Plan",
#         "Medication"
#     )

#     treatment_index = (
#         treatment_options.index(
#             treatment_default
#         )
#         if treatment_default in treatment_options
#         else 0
#     )


#     treatment_plan = st.selectbox(
#         "Treatment Plan",
#         treatment_options,
#         index=treatment_index
#     )


#     col1, col2, col3 = st.columns(3)

#     with col1:

#         treatment_duration = st.number_input(
#             "Treatment Duration (Days)",
#             min_value=1,
#             max_value=1000,
#             value=int(
#                 float(
#                     sample.get(
#                         "Treatment_Duration_Days",
#                         60
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col2:

#         medication_count = st.number_input(
#             "Medication Count",
#             min_value=0,
#             max_value=10,
#             value=int(
#                 float(
#                     sample.get(
#                         "Medication_Count",
#                         2
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col3:

#         previous_hospitalizations = st.number_input(
#             "Previous Hospitalizations",
#             min_value=0,
#             max_value=6,
#             value=int(
#                 float(
#                     sample.get(
#                         "Previous_Hospitalizations",
#                         1
#                     )
#                 )
#             ),
#             step=1
#         )


#     col1, col2, col3 = st.columns(3)

#     with col1:

#         followup_visits = st.number_input(
#             "Follow-up Visits",
#             min_value=0,
#             max_value=7,
#             value=int(
#                 float(
#                     sample.get(
#                         "Followup_Visits",
#                         2
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col2:

#         days_to_followup = st.number_input(
#             "Days to Follow-up",
#             min_value=0,
#             max_value=365,
#             value=int(
#                 float(
#                     sample.get(
#                         "Days_to_Followup",
#                         30
#                     )
#                 )
#             ),
#             step=1
#         )


#     with col3:

#         side_effect_options = [
#             "None",
#             "Mild",
#             "Moderate",
#             "Severe"
#         ]

#         side_default = sample.get(
#             "Side_Effects",
#             "None"
#         )

#         side_index = (
#             side_effect_options.index(
#                 side_default
#             )
#             if side_default in side_effect_options
#             else 0
#         )

#         side_effects = st.selectbox(
#             "Side Effects",
#             side_effect_options,
#             index=side_index
#         )


#     adherence_default = int(
#         float(
#             sample.get(
#                 "Treatment_Adherence_Percent",
#                 75
#             )
#         )
#     )

#     adherence_default = max(
#         25,
#         min(
#             100,
#             adherence_default
#         )
#     )


#     treatment_adherence = st.slider(
#         "Treatment Adherence",
#         min_value=25,
#         max_value=100,
#         value=adherence_default,
#         step=1
#     )


#     # ========================================================
#     # CREATE PATIENT DATA
#     # ========================================================

#     patient_data = {

#         "Age": age,

#         "Gender": gender,

#         "BMI": bmi,

#         "Smoking_Status":
#             smoking_status,

#         "Alcohol_Use":
#             alcohol_use,

#         "Physical_Activity":
#             physical_activity,

#         "Family_History":
#             medical_values[
#                 "Family_History"
#             ],

#         "Diabetes":
#             medical_values[
#                 "Diabetes"
#             ],

#         "Hypertension":
#             medical_values[
#                 "Hypertension"
#             ],

#         "Heart_Disease":
#             medical_values[
#                 "Heart_Disease"
#             ],

#         "Kidney_Disease":
#             medical_values[
#                 "Kidney_Disease"
#             ],

#         "Respiratory_Disease":
#             medical_values[
#                 "Respiratory_Disease"
#             ],

#         "Systolic_BP":
#             systolic_bp,

#         "Diastolic_BP":
#             diastolic_bp,

#         "Glucose_mg_dL":
#             glucose,

#         "Cholesterol_mg_dL":
#             cholesterol,

#         "Heart_Rate_bpm":
#             heart_rate,

#         "Disease_Category":
#             disease_category,

#         "Treatment_Plan":
#             treatment_plan,

#         "Treatment_Duration_Days":
#             treatment_duration,

#         "Treatment_Adherence_Percent":
#             treatment_adherence,

#         "Medication_Count":
#             medication_count,

#         "Previous_Hospitalizations":
#             previous_hospitalizations,

#         "Followup_Visits":
#             followup_visits,

#         "Days_to_Followup":
#             days_to_followup,

#         "Side_Effects":
#             side_effects
#     }


#     # ========================================================
#     # PREDICT BUTTON
#     # ========================================================

#     st.markdown("---")

#     if st.button(
#         "🔮 Predict Outcome",
#         type="primary",
#         use_container_width=True
#     ):

#         validation_errors = validate_patient(
#             patient_data
#         )

#         if validation_errors:

#             for error in validation_errors:

#                 st.error(error)

#         else:

#             try:

#                 with st.spinner(
#                     "Analyzing patient data..."
#                 ):

#                     prediction = make_prediction(
#                         patient_data
#                     )

#                     save_patient_record(
#                         patient_data,
#                         prediction
#                     )

#                     st.session_state[
#                         "prediction"
#                     ] = prediction

#                     st.session_state.pop(
#                         "sample_patient",
#                         None
#                     )


#                 if prediction.lower() == "improved":

#                     st.success(
#                         f"AI Prediction: {prediction}"
#                     )

#                 elif prediction.lower() == "stable":

#                     st.info(
#                         f"AI Prediction: {prediction}"
#                     )

#                 elif prediction.lower() == "deteriorated":

#                     st.error(
#                         f"AI Prediction: {prediction}"
#                     )

#                 else:

#                     st.warning(
#                         f"AI Prediction: {prediction}"
#                     )


#                 st.success(
#                     "Prediction saved successfully to the patient database."
#                 )


#                 # st.markdown(
#                 #     """
#                 #     <div class="info-box">
#                 #     The prediction was generated using the trained machine
#                 #     learning model and the result has been stored in the
#                 #     application database.
#                 #     </div>
#                 #     """,
#                 #     unsafe_allow_html=True
#                 # )


#             except Exception as e:

#                 st.error(
#                     f"Prediction failed: {e}"
#                 )


# # ============================================================
# # PATIENT HISTORY
# # ============================================================

# elif page == "📋 Patient History":

#     st.markdown(
#         '<div class="main-title">Patient History</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         View previously submitted patient information and their predicted health outcomes.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     if st.button(
#         "↻ Refresh"
#     ):

#         get_patient_records.clear()

#         st.rerun()


#     patients = get_patient_records()


#     if patients.empty:

#         st.info(
#             "No patient records found."
#         )

#     else:

#         # --------------------------------------------
#         # Filters
#         # --------------------------------------------

#         col1, col2, col3 = st.columns(3)


#         with col1:

#             outcome_filter = st.selectbox(
#                 "Prediction",
#                 [
#                     "All",
#                     "Improved",
#                     "Stable",
#                     "Deteriorated"
#                 ]
#             )


#         with col2:

#             diseases = [
#                 "All"
#             ] + sorted(
#                 patients[
#                     "Disease_Category"
#                 ]
#                 .dropna()
#                 .astype(str)
#                 .unique()
#                 .tolist()
#             )

#             disease_filter = st.selectbox(
#                 "Disease",
#                 diseases
#             )


#         with col3:

#             search = st.text_input(
#                 "Search Patient Records",
#                 placeholder="Search..."
#             )


#         filtered = patients.copy()


#         if outcome_filter != "All":

#             filtered = filtered[
#                 filtered["prediction"]
#                 .astype(str)
#                 .str.lower()
#                 ==
#                 outcome_filter.lower()
#             ]


#         if disease_filter != "All":

#             filtered = filtered[
#                 filtered[
#                     "Disease_Category"
#                 ].astype(str)
#                 ==
#                 disease_filter
#             ]


#         if search:

#             mask = (
#                 filtered.astype(str)
#                 .apply(
#                     lambda row:
#                     row.str.contains(
#                         search,
#                         case=False,
#                         na=False
#                     ).any(),
#                     axis=1
#                 )
#             )

#             filtered = filtered[
#                 mask
#             ]


#         # --------------------------------------------
#         # Table matching React version
#         # --------------------------------------------

#         display_df = filtered[
#             [
#                 "id",
#                 "Age",
#                 "Gender",
#                 "Disease_Category",
#                 "Treatment_Plan",
#                 "Treatment_Adherence_Percent",
#                 "prediction",
#                 "created_at"
#             ]
#         ].copy()


#         display_df.columns = [
#             "ID",
#             "Age",
#             "Gender",
#             "Disease",
#             "Treatment",
#             "Adherence",
#             "Prediction",
#             "Date"
#         ]


#         display_df["ID"] = (
#             "#"
#             +
#             display_df["ID"]
#             .astype(str)
#         )


#         display_df[
#             "Adherence"
#         ] = (
#             display_df[
#                 "Adherence"
#             ].astype(str)
#             +
#             "%"
#         )


#         st.dataframe(
#             display_df,
#             use_container_width=True,
#             hide_index=True
#         )


#         st.caption(
#             f"Showing {len(filtered)} of {len(patients)} patient records."
#         )


# # ============================================================
# # HEART DISEASE
# # ============================================================

# elif page == "❤️ Heart Disease":

#     st.markdown(
#         '<div class="main-title">Heart Disease Analysis</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         Analysis of patients with heart disease in the stored patient records.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     patients = get_patient_records()


#     if patients.empty:

#         st.info(
#             "No patient records available."
#         )

#     else:

#         heart_patients = patients[
#             patients["Heart_Disease"]
#             .astype(str)
#             .str.strip()
#             .str.lower()
#             .isin(
#                 ["yes", "1", "true"]
#             )
#         ].copy()


#         total_heart = len(
#             heart_patients
#         )


#         improved_heart = (
#             heart_patients[
#                 "prediction"
#             ]
#             .astype(str)
#             .str.lower()
#             .eq("improved")
#             .sum()
#         )


#         stable_heart = (
#             heart_patients[
#                 "prediction"
#             ]
#             .astype(str)
#             .str.lower()
#             .eq("stable")
#             .sum()
#         )


#         deteriorated_heart = (
#             heart_patients[
#                 "prediction"
#             ]
#             .astype(str)
#             .str.lower()
#             .eq("deteriorated")
#             .sum()
#         )


#         c1, c2, c3, c4 = st.columns(4)


#         c1.metric(
#             "Heart Disease Patients",
#             total_heart
#         )

#         c2.metric(
#             "Improved",
#             improved_heart
#         )

#         c3.metric(
#             "Stable",
#             stable_heart
#         )

#         c4.metric(
#             "Deteriorated",
#             deteriorated_heart
#         )


#         if total_heart == 0:

#             st.info(
#                 "No heart disease patient records are available."
#             )

#         else:

#             heart_patients[
#                 "Age"
#             ] = pd.to_numeric(
#                 heart_patients[
#                     "Age"
#                 ],
#                 errors="coerce"
#             )

#             heart_patients[
#                 "BMI"
#             ] = pd.to_numeric(
#                 heart_patients[
#                     "BMI"
#                 ],
#                 errors="coerce"
#             )

#             heart_patients[
#                 "Treatment_Adherence_Percent"
#             ] = pd.to_numeric(
#                 heart_patients[
#                     "Treatment_Adherence_Percent"
#                 ],
#                 errors="coerce"
#             )


#             c1, c2, c3 = st.columns(3)


#             c1.metric(
#                 "Average Age",
#                 f"{heart_patients['Age'].mean():.1f}"
#             )

#             c2.metric(
#                 "Average BMI",
#                 f"{heart_patients['BMI'].mean():.1f}"
#             )

#             c3.metric(
#                 "Average Adherence",
#                 f"{heart_patients['Treatment_Adherence_Percent'].mean():.1f}%"
#             )


#             outcome_df = (
#                 heart_patients[
#                     "prediction"
#                 ]
#                 .value_counts()
#                 .reset_index()
#             )

#             outcome_df.columns = [
#                 "Outcome",
#                 "Patients"
#             ]


#             fig = px.bar(
#                 outcome_df,
#                 x="Outcome",
#                 y="Patients",
#                 title="Predicted Outcomes Among Heart Disease Patients"
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )


#             st.markdown(
#                 "### Heart Disease Patient Records"
#             )


#             display = heart_patients[
#                 [
#                     "id",
#                     "Age",
#                     "Gender",
#                     "Disease_Category",
#                     "Treatment_Plan",
#                     "Treatment_Adherence_Percent",
#                     "prediction",
#                     "created_at"
#                 ]
#             ].copy()


#             display.columns = [
#                 "ID",
#                 "Age",
#                 "Gender",
#                 "Disease",
#                 "Treatment",
#                 "Adherence",
#                 "Prediction",
#                 "Date"
#             ]


#             st.dataframe(
#                 display,
#                 use_container_width=True,
#                 hide_index=True
#             )


# # ============================================================
# # DISEASE ANALYSIS
# # ============================================================

# elif page == "🦠 Disease Analysis":

#     st.markdown(
#         '<div class="main-title">Disease Analysis</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         Analyze patient distribution and predicted outcomes across disease categories.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     patients = get_patient_records()


#     if patients.empty:

#         st.info(
#             "No patient records available."
#         )

#     else:

#         disease_counts = (
#             patients[
#                 "Disease_Category"
#             ]
#             .fillna("Unknown")
#             .value_counts()
#             .reset_index()
#         )

#         disease_counts.columns = [
#             "Disease",
#             "Patients"
#         ]


#         fig = px.bar(
#             disease_counts,
#             x="Disease",
#             y="Patients",
#             title="Patients by Disease Category"
#         )

#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )


#         # ----------------------------------------
#         # Disease vs Outcome
#         # ----------------------------------------

#         cross_tab = pd.crosstab(
#             patients[
#                 "Disease_Category"
#             ],
#             patients[
#                 "prediction"
#             ]
#         ).reset_index()


#         melted = cross_tab.melt(
#             id_vars=[
#                 "Disease_Category"
#             ],
#             var_name="Outcome",
#             value_name="Patients"
#         )


#         fig = px.bar(
#             melted,
#             x="Disease_Category",
#             y="Patients",
#             color="Outcome",
#             barmode="group",
#             title="Disease Category vs Predicted Outcome"
#         )


#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )


#         # ----------------------------------------
#         # Disease selection
#         # ----------------------------------------

#         disease_list = sorted(
#             patients[
#                 "Disease_Category"
#             ]
#             .dropna()
#             .astype(str)
#             .unique()
#             .tolist()
#         )


#         selected_disease = st.selectbox(
#             "Select Disease Category",
#             disease_list
#         )


#         selected_records = patients[
#             patients[
#                 "Disease_Category"
#             ].astype(str)
#             ==
#             selected_disease
#         ]


#         st.write(
#             f"### {selected_disease} Patients"
#         )


#         st.dataframe(
#             selected_records,
#             use_container_width=True,
#             hide_index=True
#         )


# # ============================================================
# # TREATMENT ANALYSIS
# # ============================================================

# elif page == "💊 Treatment Analysis":

#     st.markdown(
#         '<div class="main-title">Treatment Analysis</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         Analyze treatment plans, adherence and predicted outcomes.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     patients = get_patient_records()


#     if patients.empty:

#         st.info(
#             "No patient records available."
#         )

#     else:

#         # ----------------------------------------
#         # Treatment distribution
#         # ----------------------------------------

#         treatment_counts = (
#             patients[
#                 "Treatment_Plan"
#             ]
#             .fillna("Unknown")
#             .value_counts()
#             .reset_index()
#         )


#         treatment_counts.columns = [
#             "Treatment",
#             "Patients"
#         ]


#         fig = px.bar(
#             treatment_counts,
#             x="Treatment",
#             y="Patients",
#             title="Treatment Plan Distribution"
#         )


#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )


#         # ----------------------------------------
#         # Treatment vs outcome
#         # ----------------------------------------

#         treatment_outcomes = pd.crosstab(
#             patients[
#                 "Treatment_Plan"
#             ],
#             patients[
#                 "prediction"
#             ]
#         ).reset_index()


#         melted = treatment_outcomes.melt(
#             id_vars=[
#                 "Treatment_Plan"
#             ],
#             var_name="Outcome",
#             value_name="Patients"
#         )


#         fig = px.bar(
#             melted,
#             x="Treatment_Plan",
#             y="Patients",
#             color="Outcome",
#             barmode="group",
#             title="Treatment Plan vs Predicted Outcome"
#         )


#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )


#         # ----------------------------------------
#         # Adherence
#         # ----------------------------------------

#         adherence = pd.to_numeric(
#             patients[
#                 "Treatment_Adherence_Percent"
#             ],
#             errors="coerce"
#         )


#         fig = px.histogram(
#             adherence,
#             nbins=10,
#             title="Treatment Adherence Distribution"
#         )


#         fig.update_layout(
#             xaxis_title="Treatment Adherence (%)",
#             yaxis_title="Number of Patients"
#         )


#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )


#         # ----------------------------------------
#         # Treatment statistics
#         # ----------------------------------------

#         c1, c2, c3 = st.columns(3)


#         c1.metric(
#             "Average Adherence",
#             f"{adherence.mean():.1f}%"
#         )


#         duration = pd.to_numeric(
#             patients[
#                 "Treatment_Duration_Days"
#             ],
#             errors="coerce"
#         )


#         c2.metric(
#             "Average Treatment Duration",
#             f"{duration.mean():.1f} days"
#         )


#         medication = pd.to_numeric(
#             patients[
#                 "Medication_Count"
#             ],
#             errors="coerce"
#         )


#         c3.metric(
#             "Average Medication Count",
#             f"{medication.mean():.1f}"
#         )


# # ============================================================
# # RISK ANALYSIS
# # ============================================================

# elif page == "⚠️ Risk Analysis":

#     st.markdown(
#         '<div class="main-title">Risk Analysis</div>',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div class="subtitle">
#         Analysis of records predicted as Deteriorated by the machine learning model.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     patients = get_patient_records()


#     if patients.empty:

#         st.info(
#             "No patient records available."
#         )

#     else:

#         risk_patients = patients[
#             patients[
#                 "prediction"
#             ]
#             .astype(str)
#             .str.lower()
#             ==
#             "deteriorated"
#         ].copy()


#         risk_count = len(
#             risk_patients
#         )


#         st.metric(
#             "Deteriorated Predictions",
#             risk_count
#         )


#         if risk_count == 0:

#             st.success(
#                 "No records are currently predicted as Deteriorated."
#             )

#         else:

#             age = pd.to_numeric(
#                 risk_patients[
#                     "Age"
#                 ],
#                 errors="coerce"
#             )

#             bmi = pd.to_numeric(
#                 risk_patients[
#                     "BMI"
#                 ],
#                 errors="coerce"
#             )

#             adherence = pd.to_numeric(
#                 risk_patients[
#                     "Treatment_Adherence_Percent"
#                 ],
#                 errors="coerce"
#             )

#             hospitalizations = pd.to_numeric(
#                 risk_patients[
#                     "Previous_Hospitalizations"
#                 ],
#                 errors="coerce"
#             )


#             c1, c2, c3, c4 = st.columns(4)


#             c1.metric(
#                 "Average Age",
#                 f"{age.mean():.1f}"
#             )

#             c2.metric(
#                 "Average BMI",
#                 f"{bmi.mean():.1f}"
#             )

#             c3.metric(
#                 "Average Adherence",
#                 f"{adherence.mean():.1f}%"
#             )

#             c4.metric(
#                 "Avg. Hospitalizations",
#                 f"{hospitalizations.mean():.1f}"
#             )


#             # ----------------------------------------
#             # Disease distribution
#             # ----------------------------------------

#             risk_diseases = (
#                 risk_patients[
#                     "Disease_Category"
#                 ]
#                 .fillna("Unknown")
#                 .value_counts()
#                 .reset_index()
#             )


#             risk_diseases.columns = [
#                 "Disease",
#                 "Patients"
#             ]


#             fig = px.bar(
#                 risk_diseases,
#                 x="Disease",
#                 y="Patients",
#                 title="Disease Distribution Among Deteriorated Predictions"
#             )


#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )


#             # ----------------------------------------
#             # Records
#             # ----------------------------------------

#             st.markdown(
#                 "### Deteriorated Patient Records"
#             )


#             st.dataframe(
#                 risk_patients,
#                 use_container_width=True,
#                 hide_index=True
#             )


# # ============================================================
# # FOOTER
# # ============================================================

# # st.markdown(
# #     """
# #     <div class="footer">
# #         Patient Health Outcome Predictor • Machine Learning Based Decision Support System
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )