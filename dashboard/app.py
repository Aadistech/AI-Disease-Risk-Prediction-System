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
    return pd.read_csv(DATA_PATH)


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

        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        return model


st.set_page_config(
    page_title="AI Healthcare Analytics System",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
# 🩺 AI-Powered Disease Risk Prediction System
### Predict diabetes risk using Machine Learning & Healthcare Analytics
""")

st.divider()

data = load_data()
model = load_model()

with st.sidebar:
    st.title("🧠 Healthcare Dashboard")
    st.markdown("AI-powered disease risk prediction system")
    st.divider()

    st.header("📊 Dataset Summary")
    st.metric("Total Rows", len(data))
    st.metric("Features", len(FEATURES))
    st.metric("Diabetic Cases", int(data["Outcome"].sum()))
    st.metric("Non-Diabetic Cases", int(len(data) - data["Outcome"].sum()))

st.subheader("📊 Health Analytics Dashboard")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Total Patients", len(data))

with col_b:
    st.metric("Diabetic Cases", int(data["Outcome"].sum()))

with col_c:
    st.metric("Non-Diabetic Cases", int(len(data) - data["Outcome"].sum()))

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Diabetes Case Distribution")
    outcome_count = data["Outcome"].value_counts()
    st.bar_chart(outcome_count)

with chart_col2:
    st.subheader("BMI vs Glucose Analysis")
    st.scatter_chart(
        data,
        x="Glucose",
        y="BMI",
        color="Outcome",
    )

st.divider()

st.subheader("🧾 Enter Patient Health Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

if st.button("🔍 Predict Risk", type="primary"):
    patient = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
        columns=FEATURES,
    )

    prediction = int(model.predict(patient)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(patient)[0][1])
        st.metric("Estimated Diabetes Risk", f"{probability:.1%}")

    if prediction == 1:
        st.error("⚠️ Prediction: Higher diabetes risk")
        st.warning("Recommendation: Please consult a doctor and monitor glucose, BMI, diet, and lifestyle habits.")
    else:
        st.success("✅ Prediction: Lower diabetes risk")
        st.info("Recommendation: Maintain a healthy diet, regular exercise, and routine health checkups.")

st.divider()

st.subheader("📄 Diabetes Dataset Preview")
st.dataframe(data.head(20), use_container_width=True)

st.divider()

st.markdown("""
<center>
Developed by <b>Aaditya Jadhav</b> 🚀 <br>
AI-Powered Healthcare Analytics Project
</center>
""", unsafe_allow_html=True)