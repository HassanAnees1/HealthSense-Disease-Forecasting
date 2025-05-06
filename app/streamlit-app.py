import streamlit as st
import pickle
import numpy as np

# تحميل النماذج
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))

st.title("Multiple Disease Prediction System")

# اختيار الصفحة
selected = st.sidebar.selectbox(
    "Select Disease Prediction",
    ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"]
)

# -------------------------------------------
# Diabetes
if selected == "Diabetes Prediction":
    st.header("Diabetes Prediction")

    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Blood Pressure value")
    SkinThickness = st.number_input("Skin Thickness value")
    Insulin = st.number_input("Insulin Level")
    BMI = st.number_input("BMI value")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value")
    Age = st.number_input("Age")

    if st.button("Predict"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = diabetes_model.predict(input_data)
        st.success("Diabetic" if result[0] == 1 else "Not Diabetic")

# -------------------------------------------
# Heart Disease
elif selected == "Heart Disease Prediction":
    st.header("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.number_input("Sex (1 = Male, 0 = Female)")
    cp = st.number_input("Chest Pain types")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholestoral")
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl")
    restecg = st.number_input("Resting Electrocardiographic results")
    thalach = st.number_input("Maximum Heart Rate")
    exang = st.number_input("Exercise Induced Angina")

    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])
        result = heart_model.predict(input_data)
        st.success("Heart Disease" if result[0] == 1 else "No Heart Disease")

# -------------------------------------------
# Parkinson's
elif selected == "Parkinson's Prediction":
    st.header("Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter = st.number_input("MDVP:Jitter(%)")
    shimmer = st.number_input("MDVP:Shimmer")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")

    if st.button("Predict"):
        input_data = np.array([[fo, fhi, flo, jitter, shimmer, nhr, hnr]])
        result = parkinsons_model.predict(input_data)
        st.success("Parkinson's Detected" if result[0] == 1 else "No Parkinson's")
