from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

DIABETES_DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
DIABETES_MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"
HEART_MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


@st.cache_data
def load_diabetes_data():
    return pd.read_csv(DIABETES_DATA_PATH)


@st.cache_resource
def load_diabetes_model():
    return joblib.load(DIABETES_MODEL_PATH)


@st.cache_resource
def load_heart_model():
    return joblib.load(HEART_MODEL_PATH)


st.set_page_config(
    page_title="AI Multi-Disease Prediction System",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
# 🩺 AI-Powered Multi-Disease Risk Prediction System
### Predict disease risk using Machine Learning & Healthcare Analytics
""")

st.divider()

disease = st.selectbox(
    "Select Disease Prediction",
    ["Diabetes", "Heart Disease"]
)

st.divider()

if disease == "Diabetes":
    data = load_diabetes_data()
    model = load_diabetes_model()

    with st.sidebar:
        st.title("🧠 Healthcare Dashboard")
        st.markdown("Diabetes prediction dashboard")
        st.divider()
        st.metric("Total Rows", len(data))
        st.metric("Features", len(DIABETES_FEATURES))
        st.metric("Diabetic Cases", int(data["Outcome"].sum()))
        st.metric("Non-Diabetic Cases", int(len(data) - data["Outcome"].sum()))

    st.subheader("📊 Diabetes Health Analytics Dashboard")

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
        st.bar_chart(data["Outcome"].value_counts())

    with chart_col2:
        st.subheader("BMI vs Glucose Analysis")
        st.scatter_chart(data, x="Glucose", y="BMI", color="Outcome")

    st.divider()

    st.subheader("🧾 Enter Diabetes Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 250, 120)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict Diabetes Risk", type="primary"):
        patient = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
            columns=DIABETES_FEATURES,
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
            st.info("Recommendation: Maintain healthy diet, regular exercise, and routine checkups.")

    st.divider()
    st.subheader("📄 Diabetes Dataset Preview")
    st.dataframe(data.head(20), use_container_width=True)


elif disease == "Heart Disease":

    heart_model = load_heart_model()

    heart_data = pd.read_csv(BASE_DIR / "data" / "heart.csv")

    with st.sidebar:
        st.title("❤️ Heart Disease Dashboard")
        st.markdown("Heart disease prediction analytics")
        st.divider()

        st.metric("Total Rows", len(heart_data))
        st.metric("Features", len(HEART_FEATURES))
        st.metric("Heart Disease Cases", int(heart_data["target"].sum()))
        st.metric(
            "No Heart Disease Cases",
            int(len(heart_data) - heart_data["target"].sum())
        )

    st.subheader("📊 Heart Disease Analytics Dashboard")

    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        st.metric("Total Patients", len(heart_data))

    with h_col2:
        st.metric("Heart Disease Cases", int(heart_data["target"].sum()))

    with h_col3:
        st.metric(
            "No Heart Disease Cases",
            int(len(heart_data) - heart_data["target"].sum())
        )

    st.divider()

    heart_chart1, heart_chart2 = st.columns(2)

    with heart_chart1:
        st.subheader("Heart Disease Case Distribution")
        st.bar_chart(heart_data["target"].value_counts())

    with heart_chart2:
        st.subheader("Age vs Cholesterol Analysis")

        st.scatter_chart(
            heart_data,
            x="age",
            y="chol",
            color="target",
        )

    st.divider()

    st.subheader("❤️ Enter Heart Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", [0, 1], help="0 = Female, 1 = Male")
        cp = st.number_input("Chest Pain Type", 0, 3, 1)
        trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.number_input("Resting ECG", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])

    with col3:
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
        slope = st.number_input("Slope", 0, 2, 1)
        ca = st.number_input("Major Vessels", 0, 4, 0)
        thal = st.number_input("Thal", 0, 3, 2)

    if st.button("❤️ Predict Heart Disease Risk", type="primary"):

        heart_patient = pd.DataFrame(
            [[
                age, sex, cp, trestbps, chol,
                fbs, restecg, thalach, exang,
                oldpeak, slope, ca, thal
            ]],
            columns=HEART_FEATURES,
        )

        heart_prediction = int(heart_model.predict(heart_patient)[0])

        if hasattr(heart_model, "predict_proba"):
            heart_probability = float(
                heart_model.predict_proba(heart_patient)[0][1]
            )

            st.metric(
                "Estimated Heart Disease Risk",
                f"{heart_probability:.1%}"
            )

        if heart_prediction == 1:
            st.error("⚠️ Prediction: Higher heart disease risk")

            st.warning(
                "Recommendation: Please consult a cardiologist and monitor BP, cholesterol, and lifestyle habits."
            )

        else:
            st.success("✅ Prediction: Lower heart disease risk")

            st.info(
                "Recommendation: Maintain heart-healthy diet, regular exercise, and routine checkups."
            )

    st.divider()

    st.subheader("📄 Heart Disease Dataset Preview")

    st.dataframe(
        heart_data.head(20),
        use_container_width=True
    )