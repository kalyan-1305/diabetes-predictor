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

st.title("🩺 Advanced Diabetes Prediction")
st.write("Enter patient data to predict the likelihood of diabetes.")

# Create columns for layout
col1, col2 = st.columns(2)

# Dictionary to map user-friendly smoking text to the number
smoking_options = {"Never Smoked": 0, "Formerly Smoked": 1, "Currently Smokes": 2}

# --- UPDATED LABELS for clarity ---
with col1:
    blood_sugar = st.number_input("Blood Sugar Level (mg/dL)", min_value=0, max_value=250, value=110)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    avg_blood_sugar = st.number_input("Avg. Blood Sugar (3-Month, %)", min_value=4.0, max_value=15.0, value=5.7, format="%.1f")

with col2:
    cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    hypertension = st.selectbox("Do you have High Blood Pressure?", ("No", "Yes"))
    heart_disease = st.selectbox("Do you have Heart Disease?", ("No", "Yes"))
    smoking_status = st.selectbox("Smoking Status", list(smoking_options.keys()))
# ---------------------------------

# Prediction button
if st.button("Predict Diabetes", use_container_width=True, type="primary"):
    
    # --- Process the inputs ---
    
    # Convert 'Yes'/'No' to 1/0
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    
    # Convert smoking text to number
    smoking_val = smoking_options[smoking_status]

    # 1. Collect input data into a 2D numpy array
    # --- CRITICAL: Must be in the SAME ORDER as the training data ---
    input_data = np.array([[
        blood_sugar,
        bmi,
        age,
        avg_blood_sugar,
        cholesterol,
        hypertension_val,
        heart_disease_val,
        smoking_val
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