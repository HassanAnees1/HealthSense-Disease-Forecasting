import streamlit as st
import pickle
import numpy as np

# ——————————————————
# تحميل الموديلات مرة واحدة
@st.cache_resource
def load_models():
    return {
        "Diabetes": pickle.load(open("models/diabetes_model.pkl","rb")),
        "Heart":    pickle.load(open("models/heart_model.pkl","rb")),
        "Parkinson":pickle.load(open("models/parkinsons_model.pkl","rb")),
    }

models = load_models()

# ——————————————————
# إعداد الصفحة
st.set_page_config(
    page_title="HealthSense Disease Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 HealthSense: Multiple Disease Prediction")

# ——————————————————
# السايدبار لاختيار المرض
disease = st.sidebar.selectbox(
    "Select Disease",
    ("Diabetes", "Heart", "Parkinson")
)

# دالة مساعدة لعمل الـ input form
def user_input(features):
    data = {}
    for name, (minv, maxv, step) in features.items():
        data[name] = st.number_input(name, min_value=minv, max_value=maxv, step=step)
    return np.array([list(data.values())])

# ——————————————————
if disease == "Diabetes":
    st.header("Diabetes Prediction")
    feat = {
        "Pregnancies": (0, 20, 1),
        "Glucose": (0, 200, 1),
        "BloodPressure": (0, 140, 1),
        "SkinThickness": (0, 100, 1),
        "Insulin": (0, 900, 1),
        "BMI": (0.0, 70.0, 0.1),
        "DiabetesPedigreeFunction": (0.0, 2.5, 0.01),
        "Age": (0, 120, 1),
    }
    X = user_input(feat)

elif disease == "Heart":
    st.header("Heart Disease Prediction")
    feat = {
        "Age": (0, 120, 1),
        "Sex (1=Male)": (0,1,1),
        "ChestPainType (0-3)": (0,3,1),
        "RestBP": (0,200,1),
        "Cholesterol": (0,600,1),
        "FastingBS (0/1)": (0,1,1),
        "RestECG (0-2)": (0,2,1),
        "MaxHR": (0,250,1),
        "ExerciseAngina (0/1)": (0,1,1),
    }
    X = user_input(feat)

else:  # Parkinson
    st.header("Parkinson's Disease Prediction")
    feat = {
        "MDVP:Fo(Hz)": (50.0, 300.0, 0.1),
        "MDVP:Fhi(Hz)": (50.0, 300.0, 0.1),
        "MDVP:Flo(Hz)": (50.0, 200.0, 0.1),
        "MDVP:Jitter(%)": (0.0, 1.0, 0.0001),
        "MDVP:Shimmer": (0.0, 10.0, 0.001),
        "NHR": (0.0, 1.0, 0.001),
        "HNR": (0.0, 50.0, 0.1),
    }
    X = user_input(feat)

# ——————————————————
# زر التنبؤ
if st.button("Predict"):
    model = models[disease]
    res = model.predict(X)[0]
    label = {
      "Diabetes": ("Not Diabetic","Diabetic"),
      "Heart":    ("No Heart Disease","Heart Disease"),
      "Parkinson":("No Parkinson's","Parkinson's Detected")
    }[disease][res]
    st.metric(f"Prediction: {disease}", label)

# ——————————————————
# قسم لعرض بعض الـ metrics أو الـ charts (اختياري)
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Model Performance"):
    st.sidebar.write("🔍 Loading evaluation metrics…")
    # ممكن هنا ترفع ملف CSV فيه دقة الموديل وتعرضه
