import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("LR_model.pkl")
columns = joblib.load("columns.pkl")   # training ke waqt save kia tha

# -------------------- UI Styling --------------------
st.set_page_config(page_title="CardioGuard", page_icon="❤️", layout="centered")

st.markdown(
    """
    <style>
        .main {background-color: #f9f9f9;}
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #ff1a1a;
            color: white;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #e63946;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Header --------------------
st.markdown("<h1 class='title'>CardioGuard 🫀</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='title'>Heart Disease & Stroke Risk Analyzer</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter patient details below to check risk of heart disease and stroke.</p>", unsafe_allow_html=True)

st.write("---")

# -------------------- Input Layout --------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👶 Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("⚧ Sex (1=Male, 0=Female)", [0, 1])
    bp = st.number_input("💉 Resting Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("🧪 Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", [0, 1])
    ekg = st.selectbox("📉 EKG Results", [0, 1])

with col2:
    max_hr = st.number_input("❤️ Max Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("🏃 Exercise Angina (1=Yes, 0=No)", [0, 1])
    st_depression = st.number_input("📊 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    thallium = st.selectbox("🔬 Thallium Test Result", [3, 6, 7])
    chest_pain = st.selectbox("💢 Chest Pain Type", [1, 2, 3, 4])
    slope = st.selectbox("📈 Slope of ST", [1, 2, 3])
    vessels = st.selectbox("🩺 Number of Vessels Fluro", [0, 1, 2, 3])

# -------------------- Data Formatting --------------------
input_dict = {
    "Age": age,
    "Sex": sex,
    "BP": bp,
    "Cholesterol": cholesterol,
    "FBS over 120": fbs,
    "EKG results": ekg,
    "Max HR": max_hr,
    "Exercise angina": exercise_angina,
    "ST depression": st_depression,
    # Thallium one-hot
    "Thallium_3": 1 if thallium == 3 else 0,
    "Thallium_6": 1 if thallium == 6 else 0,
    "Thallium_7": 1 if thallium == 7 else 0,
    # Chest pain one-hot
    "Chest pain type_1": 1 if chest_pain == 1 else 0,
    "Chest pain type_2": 1 if chest_pain == 2 else 0,
    "Chest pain type_3": 1 if chest_pain == 3 else 0,
    "Chest pain type_4": 1 if chest_pain == 4 else 0,
    # Slope one-hot
    "Slope of ST_1": 1 if slope == 1 else 0,
    "Slope of ST_2": 1 if slope == 2 else 0,
    "Slope of ST_3": 1 if slope == 3 else 0,
    # Vessels one-hot
    "Number of vessels fluro_0": 1 if vessels == 0 else 0,
    "Number of vessels fluro_1": 1 if vessels == 1 else 0,
    "Number of vessels fluro_2": 1 if vessels == 2 else 0,
    "Number of vessels fluro_3": 1 if vessels == 3 else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=columns, fill_value=0)

# -------------------- Prediction --------------------
if st.button("🔍 Predict Risk"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error("⚠️ The person has **high chances of Heart Disease  → Possible Stroke Risk😢")
    else:
        st.success("The person is **Healthy & Low Risk ☺️❤️")

    st.progress(int(proba * 100))
    st.write(f"**Probability of Heart Disease:** {proba:.2f}")
