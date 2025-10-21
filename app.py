import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or scaler files not found. Please run 'model_training.py' first.")
    st.stop()

# --- Streamlit Web App Interface ---

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient data to predict the likelihood of diabetes.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=110, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4, step=0.001, format="%.3f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

# Prediction button
if st.button("Predict Diabetes", use_container_width=True, type="primary"):
    
    # 1. Collect input data into a 2D numpy array
    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]])
    
    # 2. Scale the input data using the *loaded* scaler
    input_data_scaled = scaler.transform(input_data)
    
    # 3. Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
    # 4. Display the result
    st.divider()
    probability = prediction_proba[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f"**Prediction: Patient has Diabetes** (Probability: {probability:.2f}%)")
        st.warning("Please consult a medical professional for a formal diagnosis.")
    else:
        st.success(f"**Prediction: Patient does not have Diabetes** (Probability: {100 - probability:.2f}%)")
        st.info("This is a prediction. Regular check-ups are still recommended.")