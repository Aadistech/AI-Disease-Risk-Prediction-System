from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(r"D:\PROJECTS\AI_Disease_Risk_Prediction\data\diabetes.csv")


@st.cache_resource
def load_model() -> LogisticRegression:
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        df = load_data()
        x = df[FEATURES]
        y = df["Outcome"]
        x_train, _, y_train, _ = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        model = LogisticRegression(random_state=42)
        model.fit(x_train, y_train)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model


st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter patient health measurements to estimate diabetes risk.")

data = load_data()
model = load_model()

st.divider()

st.subheader("📊 Health Analytics Dashboard")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Total Patients", len(data))

with col_b:
    st.metric("Diabetic Cases", int(data["Outcome"].sum()))

with col_c:
    st.metric("Non-Diabetic Cases", int(len(data) - data["Outcome"].sum()))
with st.sidebar:
    st.header("Dataset")
    st.metric("Rows", len(data))
    st.metric("Features", len(FEATURES))
    st.metric("Positive cases", int(data["Outcome"].sum()))

col1, col2 = st.columns(2)

st.subheader("Diabetes Case Distribution")

outcome_count = data["Outcome"].value_counts()

st.bar_chart(outcome_count)

st.subheader("BMI vs Glucose Analysis")

st.scatter_chart(
    data,
    x="Glucose",
    y="BMI",
    color="Outcome"
)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
    blood_pressure = st.number_input("Blood pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    pedigree = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict Risk", type="primary"):
    patient = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
        columns=FEATURES,
    )
    prediction = int(model.predict(patient)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(patient)[0][1])
        st.metric("Estimated diabetes risk", f"{probability:.1%}")

    if prediction == 1:
        st.error("Prediction: Higher diabetes risk")
    else:
        st.success("Prediction: Lower diabetes risk")

st.subheader("📄 Diabetes Dataset Preview")
st.dataframe(data.head(20), use_container_width=True)
