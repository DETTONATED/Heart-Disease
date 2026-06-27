import streamlit as st
import pickle
import numpy as np

# ── Load model ──
@st.cache_resource
def load_model():
    with open('Heart_Disease.pkl', 'rb') as f:
        obj = pickle.load(f)
    return obj['model'], obj['feature_names']

model, feature_names = load_model()

# ── Page config ──
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Prediction")
st.markdown("Enter patient details below to predict the likelihood of heart disease.")
st.divider()

# ── Input form ──
col1, col2 = st.columns(2)

with col1:
    age       = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex       = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp        = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3],
                              format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[x])
    trestbps  = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=250, value=120)
    chol      = st.number_input("Serum Cholesterol (chol) mg/dl", min_value=100, max_value=600, value=200)
    fbs       = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes")
    restecg   = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2],
                              format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])

with col2:
    thalach   = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
    exang     = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak   = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope     = st.selectbox("Slope of ST Segment (slope)", options=[0, 1, 2],
                              format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    ca        = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal      = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                              format_func=lambda x: {0:"Unknown",1:"Normal",2:"Fixed Defect",3:"Reversable Defect"}[x])

st.divider()

# ── Predict ──
if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ **Heart Disease Detected**  \nConfidence: {proba[1]*100:.1f}%")
    else:
        st.success(f"✅ **No Heart Disease Detected**  \nConfidence: {proba[0]*100:.1f}%")

    st.markdown("---")
    st.caption("⚕️ This tool is for educational purposes only. Always consult a medical professional.")

# ── Sidebar ──
with st.sidebar:
    st.header("📊 About the Model")
    st.markdown("""
- **Algorithm:** Decision Tree (Gini)
- **Dataset:** UCI Heart Disease
- **Accuracy:** ~78.69%
- **Features:** 13 clinical attributes
- **Classes:** No Disease / Heart Disease
    """)
    st.markdown("---")
    st.markdown("Built with Scikit-learn & Streamlit")
